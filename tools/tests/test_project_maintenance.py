"""Exercise cleanup in a disposable workspace, never the real project tree."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        script = self.root/'tools/project_maintenance.py'
        script.parent.mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1]/'project_maintenance.py', script)
        self.script = script

    def make(self, relative, content='preserve'):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def run_tool(self, *args):
        result = subprocess.run([sys.executable, str(self.script), *args],
                                capture_output=True, text=True, check=False)
        return result.returncode, json.loads(result.stdout)

    def test_clean_requires_apply_and_preserves_data_models_and_independent_repo(self):
        cache = self.make('xarm_sdk_control/__pycache__/helper.pyc')
        retained = [self.make(p) for p in [
            'xarm_sdk_control/data/frame/depth_raw.png',
            'vision/models/__pycache__/model.pyc',
            'mujoco_xarm6/__pycache__/sim.pyc',
            '.git/__pycache__/metadata.pyc']]
        code, report = self.run_tool('clean')
        self.assertEqual(code, 0)
        self.assertFalse(report['applied'])
        self.assertTrue(cache.exists())
        code, report = self.run_tool('clean', '--apply')
        self.assertEqual(code, 0)
        self.assertTrue(report['applied'])
        self.assertFalse(cache.exists())
        self.assertTrue(all(p.exists() for p in retained))

    def evidence_index(self, content):
        p = self.make('docs/evidence/2026-10-14/result.json', content)
        self.make('docs/evidence/2026-10-14/index.json', json.dumps(dict(files=[dict(
            snapshot=p.name, sha256=hashlib.sha256(content.encode()).hexdigest())])))
        return p

    def test_audit_flags_broken_document_links_and_changed_evidence(self):
        evidence = self.evidence_index('{}')
        evidence.write_text('{"changed":true}')
        self.make('README.md', '[missing](docs/missing.md)')
        code, report = self.run_tool('audit')
        self.assertEqual(code, 1)
        self.assertEqual(report['evidence_hash_errors'], ['result.json'])
        self.assertEqual(report['broken_markdown_links'], [dict(document='README.md', target='docs/missing.md')])

    def test_audit_accepts_valid_evidence_and_links_without_modifying_inputs(self):
        evidence = self.evidence_index('{}')
        self.make('README.md', '[result](docs/evidence/2026-10-14/result.json)')
        raw = self.make('xarm_sdk_control/data/frame/depth_raw.png', 'raw')
        code, report = self.run_tool('audit')
        self.assertEqual(code, 0)
        self.assertEqual(report['evidence_hash_errors'], [])
        self.assertEqual(report['data_directories']['frame']['files'], 1)
        self.assertEqual(raw.read_text(), 'raw')
        self.assertEqual(evidence.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
