import json
from pathlib import Path
import tempfile
import unittest

from neural.train import resolve_lr_decay_steps
from training_control import TrainingControl


class TrainingContinuationTests(unittest.TestCase):
    def test_extending_target_preserves_learning_rate_horizon(self):
        self.assertEqual(resolve_lr_decay_steps(None,1000000,{'training_args':{'steps':200000}}),200000)
        self.assertEqual(resolve_lr_decay_steps(None,1000000,{'training_args':{'steps':1000000,'lr_decay_steps':200000}}),200000)
        self.assertEqual(resolve_lr_decay_steps(None,1000000),1000000)
        self.assertEqual(resolve_lr_decay_steps(500000,1000000),500000)
        for value in (0,-1,False,1.5):
            with self.assertRaises(ValueError):resolve_lr_decay_steps(value,1000000)

    def test_history_recovers_after_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cp=root/'neural/checkpoints';cp.mkdir(parents=True)
            history=cp/'training.jsonl';status=cp/'status.json'
            status.write_text(json.dumps({'state':'paused','step':100}))
            history.write_bytes(b'{"step":100,"loss":1.0}\n{"step":105,"loss":0.8}\n'+b'\x00'*400+b'{"step":101,"loss":0.9}\n')
            control=TrainingControl(root)
            self.assertEqual([r['step'] for r in control.history()],[100])
            status.write_text(json.dumps({'state':'paused','step':101}))
            self.assertEqual([r['step'] for r in control.history()],[100,101])
            with history.open('ab') as handle:handle.write(b'{"step":101,"loss":0.75}\n{"step":102,')
            self.assertEqual(control.history()[-1]['loss'],.75)
            with history.open('ab') as handle:handle.write(b'"loss":0.7}\n')
            status.write_text(json.dumps({'state':'paused','step':102}))
            self.assertEqual(control.history()[-1]['step'],102)
            history.write_bytes(b'{"step":1,"loss":2.0}\n')
            status.write_text(json.dumps({'state':'paused','step':1}))
            self.assertEqual([r['step'] for r in control.history()],[1])


if __name__=='__main__':unittest.main()
