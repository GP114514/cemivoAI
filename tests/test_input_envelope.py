from copy import deepcopy
import unittest
from interaction_policy import normalize_text_action,response_fields
from live_app import DeepSeekPlanner,LiveError
from unittest.mock import patch

class InputEnvelopeTests(unittest.TestCase):
    def test_equivalent_input_is_lossless_and_does_not_mutate_response(self):
        nested={'should_update':True,'text_action':{'action':'input','input_text':'Hello NOVA',
            'text_source':'event','text_bounds':[12,180,102,198]}}
        saved=deepcopy(nested);flat=normalize_text_action(nested)
        self.assertEqual(nested,saved)
        self.assertEqual(response_fields(flat),{'text_action':'input','input_text':'Hello NOVA','text_source':'event'})
        self.assertEqual(flat['text_bounds'],[12,180,102,198])

    def test_conflicts_unknown_fields_and_invalid_actions_are_not_accepted(self):
        for change in ({'input_text':'replacement'},{'text_bounds':[12,180,99,198]}):
            value={'text_action':{'action':'input','input_text':'Original','text_bounds':[12,180,102,198]},**change}
            with self.assertRaises(ValueError):normalize_text_action(value)
        with self.assertRaises(ValueError):normalize_text_action({'text_action':{'action':'input','execute':'tool'}})
        value=normalize_text_action({'should_update':True,'text_action':{'action':'type','input_text':'Original'}})
        with self.assertRaises(ValueError):response_fields(value)

    def test_normalization_keeps_clear_and_provider_errors_classified(self):
        planner=DeepSeekPlanner()
        value={'should_update':True,'text_action':{'action':'clear','input_text':'','text_source':'draft'}}
        with patch.object(planner,'_vision_json',return_value=value):
            self.assertEqual(response_fields(planner._toggle_vision('',{},[]))['text_action'],'clear')
        with patch.object(planner,'_vision_json',return_value={'text_action':{'action':'send','extra':True}}):
            with self.assertRaises(LiveError) as caught:planner._toggle_vision('',{},[])
            self.assertEqual(caught.exception.code,'invalid_toggle_plan')

if __name__=='__main__':unittest.main()
