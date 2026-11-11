"""Camera-referenced encoder-to-board calibration with revolute-axis lines.

Uses the standard product-of-exponentials model. Each line has two direction
and two transverse-position parameters, plus a six-parameter home board pose.
No claim to identify a unique robot-base frame, TCP, or individual DH errors.
Inspired by photogrammetric robot calibration, not a full paper reproduction.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .station_tying import pose


def transverse_basis(n):
    n=np.asarray(n,float)/np.linalg.norm(n)
    a=np.eye(3)[np.argmin(abs(n))];a-=n*(n @a);a/=np.linalg.norm(a)
    return np.c_[a,np.cross(n,a)]


class CameraPOE:
    def __init__(self,axes,points,home,parameters=None):
        self.axes,self.points,self.home=np.asarray(axes),np.asarray(points),np.asarray(home)
        self.bases=np.array([transverse_basis(n) for n in self.axes])
        self.parameters=np.zeros(30) if parameters is None else np.asarray(parameters)

    def decode(self,x):
        axis_delta=np.asarray(x[6:]).reshape(6,4)
        directions=Rotation.from_rotvec(np.einsum('nij,nj->ni',self.bases,axis_delta[:,:2]*.005)).apply(self.axes)
        points=self.points+np.einsum('nij,nj->ni',self.bases,axis_delta[:,2:]*.001)
        points-=np.sum(points*directions,axis=1)[:,None]*directions
        home=pose(Rotation.from_rotvec(x[3:6]*.1).as_matrix() @self.home[:3,:3],self.home[:3,3]+x[:3]*.01)
        return directions,points,home

    def forward(self,q,x=None):
        qs=np.atleast_2d(q);n,p,H=self.decode(self.parameters if x is None else x)
        R=np.tile(np.eye(3),(len(qs),1,1));t=np.zeros((len(qs),3))
        for i in range(6):
            r=Rotation.from_rotvec(qs[:,i,None]*n[i]).as_matrix()
            shift=p[i]-np.einsum('nij,j->ni',r,p[i])
            t+=np.einsum('nij,nj->ni',R,shift);R=R @r
        out=np.tile(np.eye(4),(len(qs),1,1));out[:,:3,:3]=R @H[:3,:3]
        out[:,:3,3]=t+np.einsum('nij,j->ni',R,H[:3,3])
        return out if np.asarray(q).ndim==2 else out[0]

    def as_dict(self):
        return dict(axes=self.axes,points=self.points,home=self.home,parameters=self.parameters)


def fit_camera_poe(q,pixels,board,K,nominal_axes_B,nominal_points_B,nominal_B_T_F0,initial_X,initial_Y,*,block_robust=False):
    X=np.array(initial_X);R=X[:3,:3].T
    axes=np.asarray(nominal_axes_B) @R.T
    points=(np.asarray(nominal_points_B)-X[:3,3]) @R.T
    points-=np.sum(points*axes,axis=1)[:,None]*axes
    model=CameraPOE(axes,points,np.linalg.inv(X) @nominal_B_T_F0 @initial_Y)
    pixels,board=np.asarray(pixels),np.asarray(board)
    image_count=pixels.size
    def residual(x):
        T=model.forward(q,x)
        p=np.einsum('nij,kj->nki',T[:,:3,:3],board)+T[:,None,:3,3]
        uv=p[:,:,:2]/p[:,:,2,None]*[K[0,0],K[1,1]]+K[:2,2]
        return np.r_[((uv-pixels)/.15).ravel(),x[6:]]
    def loss(z):
        rho=np.vstack([z,np.ones_like(z),np.zeros_like(z)])
        ids=np.flatnonzero(z[:image_count]>4);root=np.sqrt(z[ids])
        rho[0,ids]=4*root-4;rho[1,ids]=2/root;rho[2,ids]=-1/(z[ids]*root)
        return rho
    fitted=least_squares(residual,np.zeros(30),loss=loss,max_nfev=100,
        ftol=1e-9,xtol=1e-9,gtol=1e-7,x_scale='jac')
    evaluations=fitted.nfev
    view_weights=np.ones(len(q));weight_history=[]
    if block_robust:
        # Cauchy loss on one board-view block. A coherent image displacement is
        # not treated as 70 independent corner outliers. Standard IRLS, no GT,
        # no frame-ID or residual threshold selected from evaluation error.
        for iteration in range(6):
            r=residual(fitted.x)[:image_count].reshape(len(q),-1)
            view_weights=1/(1+np.mean(r*r,axis=1)/4.)
            weight_history.append(view_weights.copy())
            scale=np.repeat(np.sqrt(view_weights),r.shape[1])
            def weighted(x):
                r=residual(x);return np.r_[r[:image_count]*scale,r[image_count:]]
            refined=least_squares(weighted,fitted.x,max_nfev=70,
                ftol=1e-9,xtol=1e-9,gtol=1e-7,x_scale='jac')
            evaluations+=refined.nfev
            change=np.linalg.norm(refined.x-fitted.x);fitted=refined
            if change<1e-5:break
    model.parameters=fitted.x
    J=fitted.jac[:image_count]
    singular=np.linalg.svd(J,compute_uv=False)
    return dict(accepted=bool(fitted.success),reason='camera_joint_mapping_candidate_not_robot_motion_approval',
        camera_poe=model.as_dict(),pixel_rms=float(np.sqrt(np.mean((residual(fitted.x)[:image_count]*.15)**2))),
        evaluations=evaluations,data_jacobian_singular_values=singular,
        block_robust=block_robust,view_weights=view_weights,view_weight_history=weight_history,
        regularized_axis_line_parameters=True,reported_DH_or_base_parameters_not_identified=True,
        encoder_scale_fixed_at_nominal=True,covariance_not_calibrated=True)


def solve_camera_target(model,target,seed,limits):
    """Bounded offline IK in camera coordinates, no controller access."""
    def residual(q):
        actual=model.forward(q)
        angle=Rotation.from_matrix(target[:3,:3] @actual[:3,:3].T).as_rotvec()
        return np.r_[actual[:3,3]-target[:3,3],angle*.15]
    fit=least_squares(residual,np.clip(seed,limits[:,0]+1e-8,limits[:,1]-1e-8),bounds=(limits[:,0],limits[:,1]),
        ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=100)
    p=residual(fit.x)
    return dict(q_rad=fit.x,passed=bool(np.linalg.norm(p[:3])<.0001 and np.linalg.norm(p[3:])/.15<np.deg2rad(.05)))
