import unittest
from computer_control import ComputerControl,validate_action
from windows_host import HostError
from windows_settings import WindowsSettings


class Host:
    available=True
    def info(self):return {'connected':True,'name':'TEST-PC'}
    def wifi(self):return {'available':True,'profiles':[],'interfaces':[],'networks':[],'errors':[]}
    error=staticmethod(lambda error:{'code':error.code,'message':str(error)})

class Settings:
    def __init__(self):self.volume=42;self.muted=False
    def read(self,page,inquiry=False):
        if page=='sound':return {'available':True,'volume':self.volume,'muted':self.muted,'controls':['volume','mute']}
        return {'available':True,'controls':['refresh']}
    def set_audio(self,level=None,muted=None):
        if level is not None:self.volume=level
        if muted is not None:self.muted=muted
        return {'ok':True,'status':'verified','sound':self.read('sound')}

class SettingsHubTests(unittest.TestCase):
    def setUp(self):self.control=ComputerControl(Host());self.control.settings=Settings()

    def test_settings_home_exposes_all_categories_without_becoming_wifi(self):
        result=self.control.invoke({'action':'settings','page':'home'},{},{'text':'Settings'})
        self.assertEqual(result['page'],'settings')
        self.assertEqual({i['label'] for i in result['system_state']['settings_categories']},
            {'Wi-Fi','Sound','Display','Bluetooth','Devices','About'})

    def test_generic_settings_cannot_be_silently_routed_to_wifi(self):
        value={'system_action':{'action':'settings','page':'wifi'},'requires_tools':True,
            'should_update':False,'update_mode':'none','target_type':'other','next_prompt':''}
        self.assertTrue(self.control.needs_routing(value,{}, {'text':'打开设置'}))
        value['system_action']['page']='home'
        self.assertFalse(self.control.needs_routing(value,{}, {'text':'打开设置'}))

    def test_audio_values_are_verified_and_render_the_sound_page(self):
        result=self.control.invoke({'action':'audio_volume','value':37},{},{'text':'Volume 37'})
        self.assertTrue(result['ok']);self.assertEqual(result['page'],'sound')
        self.assertEqual(result['system_state']['sound']['volume'],37)
        result=self.control.invoke({'action':'audio_mute','value':True},{},{'text':'Mute'})
        self.assertTrue(result['system_state']['sound']['muted'])

    def test_other_categories_keep_their_own_page_identity(self):
        for page in ('sound','display','bluetooth','devices'):
            result=self.control.invoke({'action':'settings','page':page},{},{})
            self.assertEqual(result['page'],page);self.assertIn(page,result['system_state'])

    def test_audio_arguments_are_strict_before_native_access(self):
        for action in ({'action':'audio_volume','value':True},{'action':'audio_volume','value':101},
                {'action':'audio_mute','value':'yes'}):
            with self.assertRaises(ValueError):validate_action(action)
        with self.assertRaises(HostError):WindowsSettings().set_audio(level=-1)

if __name__=='__main__':unittest.main()
