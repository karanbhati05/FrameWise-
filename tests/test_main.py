import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main

class Checks(unittest.TestCase):
    def test_audio_quality(self):
        rate=16000; t=np.arange(rate*2)/rate; clean=.3*np.sin(2*np.pi*220*t); clean[t<.4]=0; noisy=clean+np.random.default_rng(2).normal(0,.1,len(t)); enhanced=main.enhance(noisy,rate); self.assertEqual(len(noisy),len(enhanced)); self.assertTrue(np.isfinite(enhanced).all()); self.assertGreater(main.snr(clean,enhanced),main.snr(clean,noisy))
    def test_retrieval_abstains(self):
        self.assertEqual(main.grounded_brief([{'start':0,'end':1,'text':'Audio quality'}],'football'),[])

if __name__=="__main__": unittest.main()
