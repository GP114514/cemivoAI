from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from interaction_policy import visual_state
from live_app import LiveApp,LiveError
from neural.message_window import compile_window


def png(pixels):
    result=BytesIO();Image.fromarray(pixels).save(result,format='PNG');return result.getvalue()


class MessageWindowTests(unittest.TestCase):
    def setUp(self):
        self.area=[12,76,244,180]
        self.state={'sent_messages':['Message '+str(i) for i in range(1,6)],
            'sent_message_bounds':[[18,y,244,y+18] for y in (82,102,122,142,162)],
            'input_text':'Message 6','input_bounds':[12,194,172,212]}
        self.pixels=np.full((256,256),255,dtype=np.uint8)
        for x,y,_,_ in self.state['sent_message_bounds']:self.pixels[y+3:y+12,x+2:x+70]=0

    def test_overflow_retains_history_and_shows_newest_messages_in_order(self):
        before=deepcopy(self.state)
        window=compile_window(png(self.pixels),self.state,self.area,'Message 6')
        self.assertEqual(self.state,before)
        self.assertEqual([i['text'] for i in window['items']],['Message '+str(i) for i in range(2,7)])
        state=visual_state(self.state,{'text_action':'send','input_text':'Message 6'},
            {'text_applied':True,'preview':False,'message_window':window,'text_bounds':window['items'][-1]['bounds']})
        self.assertEqual(state['sent_messages'],['Message '+str(i) for i in range(1,7)])
        self.assertEqual(state['visible_sent_messages'],state['sent_messages'][1:])
        self.assertEqual(len(state['sent_message_bounds']),5)
        self.assertEqual(state['input_text'],'')
        self.assertNotIn('input_bounds',state)

    def test_untracked_pixels_and_unaligned_history_cannot_be_erased(self):
        self.pixels[101,22]=0
        with self.assertRaisesRegex(ValueError,'untracked'):
            compile_window(png(self.pixels),self.state,self.area,'Message 6')
        self.state['visible_sent_messages']=['wrong history']
        with self.assertRaisesRegex(ValueError,'aligned'):
            compile_window(png(self.pixels),self.state,self.area,'Message 6')

    def test_unsigned_scroll_never_reaches_the_pixel_renderer(self):
        with TemporaryDirectory() as folder:
            live=LiveApp(folder,renderer=object())
            with self.assertRaises(LiveError) as caught:
                live._draw_region(png(self.pixels),{},
                    {'text_action':'send','requires_tools':False,'_message_window':{}},{'type':'click'})
            self.assertEqual(caught.exception.code,'message_window_unverified')

    def test_live_status_reports_a_dead_training_process(self):
        with TemporaryDirectory() as folder:
            cp=Path(folder)/'neural/checkpoints';cp.mkdir(parents=True)
            (cp/'status.json').write_text(json.dumps({'state':'running','step':222320,'pid':12345,'total_steps':1000000}))
            live=LiveApp(folder,renderer=object())
            with patch('training_control.process_alive',return_value=False):
                status=live.status()
            self.assertEqual(status['training']['state'],'failed')
            self.assertEqual(status['training']['step'],222320)


if __name__=='__main__':unittest.main()
