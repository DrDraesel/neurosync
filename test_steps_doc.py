"""docs/STEPS.md contract test: the operator-steps doc must exist and keep its
required sections, so the documentation cannot rot silently.

No hardware: this only reads the file and asserts the sections are present.
"""
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent
STEPS = BASE / 'docs' / 'STEPS.md'


class TestStepsDoc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing file fails loudly (FileNotFoundError) instead of skipping:
        # the doc is part of the contract and may not rot.
        cls.text = STEPS.read_text(encoding='utf-8')

    def test_doc_is_substantial(self):
        self.assertGreater(len(self.text), 3000)

    def test_brainbit_classic_steps_and_no_calibration(self):
        self.assertIn('BrainBit Classic', self.text)
        self.assertIn('Power the headset on', self.text)
        self.assertIn('Scan devices', self.text)
        self.assertIn('Connect', self.text)
        self.assertIn('Check electrodes', self.text)
        # the SDK has no Classic calibration step; the app verifies 250 Hz
        self.assertIn('NO calibration step', self.text)
        self.assertIn('250 Hz', self.text)

    def test_dragon_pairing_and_amplifier_readback(self):
        self.assertIn('DragonEEG', self.text)
        self.assertIn('pairing mode', self.text)
        self.assertIn('WRITES the amplifier parameter block', self.text)
        self.assertIn('READS IT BACK', self.text)
        self.assertIn('mandatory pre-stream configuration step', self.text)

    def test_brainbit2_parameter_block(self):
        self.assertIn('BrainBit 2 / Pro / Flex', self.text)
        self.assertIn('amplifier parameter block', self.text)

    def test_connection_loss_recovery_order(self):
        self.assertIn('connection drops', self.text.lower())
        self.assertIn('Auto-reconnect first', self.text)
        self.assertIn('power-cycle', self.text)

    def test_study_record_flow(self):
        self.assertIn('Set subject', self.text)
        self.assertIn('Start study', self.text)
        self.assertIn('Stop', self.text)
        self.assertIn('AI analysis', self.text)
        self.assertIn('Patient report', self.text)
        self.assertIn('recorded_span_s', self.text)

    def test_honesty_list_present(self):
        self.assertIn('What this data can and cannot support', self.text)
        self.assertIn('The numbers can support', self.text)
        self.assertIn('The numbers cannot support', self.text)
        self.assertIn('not a medical device', self.text.lower())
        self.assertIn('Interpretation is clinician-led', self.text)


if __name__ == '__main__':
    unittest.main()
