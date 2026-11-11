"""Observation-error hand-eye calibration, stationary camera, SI units.

Adaptation of Ulrich & Hillemann's fictitious-observation model (TRO 2023,
Sec. III-F). Uses local rotation-vector charts and fixed pinhole intrinsics.
It is not their full HALCON implementation or kinematic robot calibration.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .fixed_camera_calibration import solve_eye_to_hand
from .station_tying import pose


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def left_jacobian(r):
    theta = np.linalg.norm(r)
    S = skew(r)
    if theta < 1e-7:
        return np.eye(3)+.5*S+S @ S/6
    return np.eye(3)+(1-np.cos(theta))/theta**2*S+(theta-np.sin(theta))/theta**3*S @ S


class HandEyeResidual:
    """Pixel and reported-pose residuals with analytic sparse-block Jacobian."""
    def __init__(self, robot, pixels, points, K, X0, Y0, sigmas):
        self.robot, self.pixels = np.asarray(robot), np.asarray(pixels)
        self.points, self.K = np.asarray(points), np.asarray(K)
        self.X0, self.Y0 = np.asarray(X0), np.asarray(Y0)
        self.n, self.m = self.pixels.shape[:2]
        self.size = 12+6*self.n
        self.scales = np.r_[np.tile([.01,.01,.01,.1,.1,.1],2),np.tile([.001,.001,.001,.005,.005,.005],self.n)]
        self.sigmas = np.asarray(sigmas, float)  # pixel, translation metre, rotation rad
        self.groups = [slice(0,2*self.n*self.m),slice(2*self.n*self.m,2*self.n*self.m+3*self.n),slice(2*self.n*self.m+3*self.n,None)]

    def decode(self, x):
        u = np.asarray(x)*self.scales
        X = pose(Rotation.from_rotvec(u[3:6]).as_matrix() @ self.X0[:3,:3], self.X0[:3,3]+u[:3])
        Y = pose(Rotation.from_rotvec(u[9:12]).as_matrix() @ self.Y0[:3,:3], self.Y0[:3,3]+u[6:9])
        delta = u[12:].reshape(self.n,6)
        A = self.robot.copy()
        A[:,:3,:3] = Rotation.from_rotvec(delta[:,3:]).as_matrix() @ A[:,:3,:3]
        A[:,:3,3] += delta[:,:3]
        return X,Y,A,u,delta

    def evaluate(self, x, jacobian=False):
        X,Y,A,u,delta = self.decode(x)
        object_rotated = self.points @ Y[:3,:3].T
        tool_points = object_rotated+Y[:3,3]
        world_rotated = np.einsum('nij,kj->nki',A[:,:3,:3],tool_points)
        world = world_rotated+A[:,None,:3,3]
        camera = (world-X[:3,3]) @ X[:3,:3]
        if np.any(camera[:,:,2] <= .01):
            raise ValueError('board projects behind camera during calibration')
        uv = camera[:,:,:2]/camera[:,:,2,None]
        uv = uv*np.array([self.K[0,0],self.K[1,1]])+self.K[:2,2]
        residual = np.r_[((uv-self.pixels)/self.sigmas[0]).ravel(),
                         (delta[:,:3]/self.sigmas[1]).ravel(),(delta[:,3:]/self.sigmas[2]).ravel()]
        if not jacobian:
            return residual
        J = np.zeros((len(residual),self.size))
        fx,fy = self.K[0,0],self.K[1,1]
        LX,LY = left_jacobian(u[3:6]),left_jacobian(u[9:12])
        invR = X[:3,:3].T
        for i in range(self.n):
            z = camera[i,:,2]
            D = np.zeros((self.m,2,3))
            D[:,0,0],D[:,1,1] = fx/z,fy/z
            D[:,0,2],D[:,1,2] = -fx*camera[i,:,0]/z**2,-fy*camera[i,:,1]/z**2
            block = np.empty((self.m,3,18))
            block[:,:,:3] = -invR
            block[:,:,3:6] = np.array([invR @ skew(v-X[:3,3]) @ LX for v in world[i]])
            block[:,:,6:9] = invR @ A[i,:3,:3]
            block[:,:,9:12] = np.array([-invR @ A[i,:3,:3] @ skew(v) @ LY for v in object_rotated])
            block[:,:,12:15] = invR
            LA = left_jacobian(delta[i,3:])
            block[:,:,15:18] = np.array([-invR @ skew(v) @ LA for v in world_rotated[i]])
            image_J = np.einsum('nki,nij->nkj',D,block).reshape(2*self.m,18)/self.sigmas[0]
            start = i*2*self.m
            J[start:start+2*self.m,:12] = image_J[:,:12]
            J[start:start+2*self.m,12+6*i:18+6*i] = image_J[:,12:]
            trow = 2*self.n*self.m+3*i
            rrow = 2*self.n*self.m+3*self.n+3*i
            J[trow:trow+3,12+6*i:15+6*i] = np.eye(3)/self.sigmas[1]
            J[rrow:rrow+3,15+6*i:18+6*i] = np.eye(3)/self.sigmas[2]
        return J*self.scales


def solve_uncertain_eye_to_hand(robot, pixels, points, K, *, variance_components=True,
                               initial_sigmas=(.1,.001,np.deg2rad(.1)), max_outer=6):
    """No truth inputs. Returned covariance is conditional on the error model.

    Corrected training robot poses are diagnostic, not commands or a validated
    general-purpose forward-kinematics correction for unseen poses.
    """
    robot, pixels = np.asarray(robot,float),np.asarray(pixels,float)
    initial = solve_eye_to_hand(robot,pixels,points,K,method='park')
    if not initial['accepted']:
        return initial
    model = HandEyeResidual(robot,pixels,points,K,initial['B_T_C'],initial['F_T_P'],initial_sigmas)
    x, history = np.zeros(model.size), []
    total_evaluations = 0
    for outer in range(max_outer if variance_components else 1):
        fit = least_squares(model.evaluate,x,jac=lambda v:model.evaluate(v,True),
            max_nfev=70,ftol=1e-9,xtol=1e-9,gtol=1e-7)
        x = fit.x
        total_evaluations += fit.nfev
        J = model.evaluate(x,True)
        inv_info = np.linalg.pinv(J.T @ J,rcond=1e-12,hermitian=True)
        redundancy = np.maximum(0,1-np.sum((J @ inv_info)*J,axis=1))
        residual = model.evaluate(x)
        factors = np.array([np.sqrt(float(residual[g] @ residual[g])/max(float(redundancy[g].sum()),1e-9)) for g in model.groups])
        history.append(dict(sigmas=model.sigmas.copy(),standard_deviation_factors=factors,
                            redundant_observations=[float(redundancy[g].sum()) for g in model.groups],
                            evaluations=fit.nfev,optimizer_success=bool(fit.success)))
        if not variance_components or np.max(abs(np.log(np.maximum(factors,1e-12)))) < .05:
            break
        if outer+1 < max_outer:
            model.sigmas = np.maximum([1e-4,1e-7,1e-7],model.sigmas*np.clip(factors,.25,4))
    X,Y,A,u,delta = model.decode(x)
    covariance = inv_info[:12,:12]*model.scales[:12,None]*model.scales[None,:12]
    rechart = np.eye(12)
    rechart[3:6,3:6] = left_jacobian(u[3:6])
    rechart[9:12,9:12] = left_jacobian(u[9:12])
    covariance = rechart @covariance @rechart.T
    pixel = model.evaluate(x)[model.groups[0]]*model.sigmas[0]
    return dict(accepted=bool(fit.success),reason='calibration_candidate_not_process_release',
        B_T_C=X,F_T_P=Y,corrected_training_B_T_F=A,
        robot_corrections_m_rad=delta,pixel_rms=float(np.sqrt(np.mean(pixel**2))),
        estimated_sigmas=dict(pixel=float(model.sigmas[0]),translation_m=float(model.sigmas[1]),rotation_rad=float(model.sigmas[2])),
        variance_components=variance_components,history=history,evaluations=total_evaluations,
        joint_X_Y_covariance=covariance,joint_covariance_chart='X/Y translation and left-local rotation at estimated transforms',
        covariance_excludes_fixed_bias_and_model_mismatch=True,
        training_pose_correction_is_not_robot_kinematic_calibration=True)
