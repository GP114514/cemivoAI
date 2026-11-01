from copy import deepcopy
from io import BytesIO
import numpy as np
from PIL import Image
import unittest
from neural.layout_surface import validate_layout,text_interior,verify_borders


class NeuralLayoutTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary=set(' ChatSend')
        self.layout={'screen_type':'chat','elements':[
            {'kind':'text','bounds':[8,36,248,54],'text':'Chat'},
            {'kind':'panel','bounds':[8,60,248,172],'text':''},
            {'kind':'input','bounds':[8,180,168,212],'text':''},
            {'kind':'button','bounds':[176,180,248,212],'text':'Send'}]}

    def test_ai_layout_is_retained_without_invented_controls(self):
        self.assertEqual(validate_layout(self.layout,self.vocabulary),self.layout)

    def test_invalid_geometry_cannot_cover_chrome_or_other_controls(self):
        for bounds in ([8,30,248,54],[8,36,260,54],[8,36,False,54],[8,36,248,63]):
            value=deepcopy(self.layout);value['elements'][0]['bounds']=bounds
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):
                validate_layout(value,self.vocabulary)
        value=deepcopy(self.layout);value['elements'][2]['bounds']=[10,70,166,104]
        with self.assertRaises(ValueError):validate_layout(value,self.vocabulary)

    def test_unknown_glyphs_overflow_and_untrained_controls_are_rejected(self):
        for text in ('ChatChatChatChatChatChatChatChatChat','未知'):
            value=deepcopy(self.layout);value['elements'][0]['text']=text
            with self.subTest(text=text),self.assertRaises(ValueError):validate_layout(value,self.vocabulary)
        value=deepcopy(self.layout);value['elements'][3]['kind']='toggle'
        with self.assertRaises(ValueError):validate_layout(value,self.vocabulary)

    def test_text_cannot_touch_input_border_or_another_control(self):
        frame={'generator':'dsh_layout_neural_surface','neural_layout':self.layout}
        self.assertEqual(text_interior(frame,'input',[12,184,120,202]),[12,184,164,208])
        for bounds in ([13,198,121,216],[180,184,240,202]):
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):text_interior(frame,'input',bounds)
        self.assertEqual(text_interior(frame,'send',[12,64,120,82]),[12,64,244,168])

    def test_border_comparison_rejects_a_changed_outline(self):
        def png(pixels):
            output=BytesIO();Image.fromarray(pixels).save(output,format='PNG');return output.getvalue()
        frame={'generator':'dsh_layout_neural_surface','neural_layout':self.layout}
        pixels=np.full((256,256),255,dtype=np.uint8);before=png(pixels)
        pixels[184,12]=0;verify_borders(before,png(pixels),frame)
        pixels[190,8]=0
        with self.assertRaises(ValueError):verify_borders(before,png(pixels),frame)


if __name__=='__main__':unittest.main()
