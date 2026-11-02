"""Native settings facts and endpoint volume; no external settings windows."""
from contextlib import contextmanager
import ctypes as C
from ctypes import wintypes as W
import hashlib
import sys
import uuid

from windows_host import GUID,HostError

def guid(value):return GUID.from_buffer_copy(uuid.UUID(value).bytes_le)
def hr(value):
    if value<0:raise HostError('windows_settings_error','Windows 系统接口未完成操作。',value & 0xffffffff)

def method(pointer,index,result,*arguments):
    table=C.cast(pointer,C.POINTER(C.POINTER(C.c_void_p))).contents
    return C.WINFUNCTYPE(result,C.c_void_p,*arguments)(table[index])

class PropertyKey(C.Structure):_fields_=[('fmtid',GUID),('pid',W.DWORD)]
class PropertyValue(C.Structure):_fields_=[('vt',W.WORD),('reserved',W.WORD*3),('data',C.c_byte*16)]

@contextmanager
def endpoint():
    if sys.platform!='win32':raise HostError('unsupported_platform','声音控制仅支持本机 Windows。')
    ole=C.WinDLL('ole32');ole.CoInitializeEx.argtypes=[C.c_void_p,W.DWORD];ole.CoInitializeEx.restype=C.c_long
    initialized=ole.CoInitializeEx(None,2);needs_cleanup=initialized in (0,1)
    if initialized<0 and initialized!=-2147417850:hr(initialized)
    ole.CoCreateInstance.argtypes=[C.POINTER(GUID),C.c_void_p,W.DWORD,C.POINTER(GUID),C.POINTER(C.c_void_p)]
    ole.CoCreateInstance.restype=C.c_long
    enum=C.c_void_p();device=C.c_void_p();volume=C.c_void_p();name='Default output'
    try:
        hr(ole.CoCreateInstance(C.byref(guid('bcde0395-e52f-467c-8e3d-c4579291692e')),None,23,
            C.byref(guid('a95664d2-9614-4f35-a746-de8db63617e6')),C.byref(enum)))
        hr(method(enum,4,C.c_long,C.c_int,C.c_int,C.POINTER(C.c_void_p))(enum,0,1,C.byref(device)))
        hr(method(device,3,C.c_long,C.POINTER(GUID),W.DWORD,C.c_void_p,C.POINTER(C.c_void_p))(
            device,C.byref(guid('5cdf2c82-841e-4546-9722-0cf74078229a')),23,None,C.byref(volume)))
        store=C.c_void_p();value=PropertyValue()
        try:
            hr(method(device,4,C.c_long,W.DWORD,C.POINTER(C.c_void_p))(device,0,C.byref(store)))
            key=PropertyKey(guid('a45c254e-df1c-4efd-8020-67d146a850e0'),14)
            hr(method(store,5,C.c_long,C.POINTER(PropertyKey),C.POINTER(PropertyValue))(store,C.byref(key),C.byref(value)))
            if value.vt==31:name=C.cast(C.byref(value,8),C.POINTER(C.c_wchar_p)).contents.value or name
        except HostError:pass
        finally:
            if value.vt:
                ole.PropVariantClear.argtypes=[C.POINTER(PropertyValue)];ole.PropVariantClear(C.byref(value))
            if store:method(store,2,W.ULONG)(store)
        yield volume,name
    finally:
        for pointer in (volume,device,enum):
            if pointer:method(pointer,2,W.ULONG)(pointer)
        if needs_cleanup:ole.CoUninitialize()

def audio_state(volume,name):
    level=C.c_float();muted=W.BOOL()
    hr(method(volume,9,C.c_long,C.POINTER(C.c_float))(volume,C.byref(level)))
    hr(method(volume,15,C.c_long,C.POINTER(W.BOOL))(volume,C.byref(muted)))
    return {'available':True,'output_name':name,'volume':round(level.value*100),'muted':bool(muted.value),
        'source':'Windows Core Audio','controls':['volume','mute']}

class DeviceMode(C.Structure):
    _fields_=([('device',W.WCHAR*32),('spec',W.WORD),('driver',W.WORD),('size',W.WORD),('extra',W.WORD),
        ('fields',W.DWORD),('position',C.c_byte*16),('color',C.c_short*5),('form',W.WCHAR*32),('pixels',W.WORD)]+
        [(name,W.DWORD) for name in ('bits','width','height','flags','frequency','icm_method','icm_intent',
            'media','dither','reserved1','reserved2','panning_width','panning_height')])

class RadioParams(C.Structure):_fields_=[('size',W.DWORD)]
class RadioInfo(C.Structure):
    _fields_=[('size',W.DWORD),('address',C.c_ulonglong),('name',W.WCHAR*248),('device_class',W.ULONG),
        ('version',W.USHORT),('manufacturer',W.USHORT)]
class SystemTime(C.Structure):_fields_=[('values',W.WORD*8)]
class BluetoothInfo(C.Structure):
    _fields_=[('size',W.DWORD),('address',C.c_ulonglong),('device_class',W.ULONG),('connected',W.BOOL),
        ('remembered',W.BOOL),('authenticated',W.BOOL),('seen',SystemTime),('used',SystemTime),('name',W.WCHAR*248)]
class SearchParams(C.Structure):
    _fields_=[('size',W.DWORD),('authenticated',W.BOOL),('remembered',W.BOOL),('unknown',W.BOOL),
        ('connected',W.BOOL),('inquiry',W.BOOL),('timeout',W.BYTE),('radio',W.HANDLE)]
class DeviceInfo(C.Structure):
    _fields_=[('size',W.DWORD),('class_guid',GUID),('instance',W.DWORD),('reserved',C.c_size_t)]


class WindowsSettings:
    def read(self,page,inquiry=False):
        try:
            if sys.platform!='win32':raise HostError('unsupported_platform','此系统功能仅支持 Windows。')
            if page=='sound':
                with endpoint() as (volume,name):return audio_state(volume,name)
            if page=='display':
                user=C.WinDLL('user32');user.EnumDisplaySettingsW.argtypes=[W.LPCWSTR,W.DWORD,C.POINTER(DeviceMode)]
                user.EnumDisplaySettingsW.restype=W.BOOL
                mode=DeviceMode();mode.size=C.sizeof(mode)
                if not user.EnumDisplaySettingsW(None,0xffffffff,C.byref(mode)):raise HostError('display_unavailable','无法读取当前显示模式。')
                return {'available':True,'width':mode.width,'height':mode.height,'refresh_hz':mode.frequency,
                    'color_bits':mode.bits,'source':'Windows display API','controls':['refresh'],
                    'note':'Current display information; resolution changes are not enabled.'}
            if page=='bluetooth':return self.bluetooth(inquiry)
            if page=='devices':return self.devices()
            raise HostError('unknown_settings_page','未知系统分类。')
        except (HostError,OSError) as error:
            return {'available':False,'error':{'code':getattr(error,'code','windows_error'),'message':str(error)},'controls':[]}

    def set_audio(self,level=None,muted=None):
        if level is not None and (type(level) is not int or not 0<=level<=100):raise HostError('invalid_volume','音量范围为 0–100。')
        if muted is not None and type(muted) is not bool:raise HostError('invalid_mute','静音状态必须是布尔值。')
        with endpoint() as (volume,name):
            before=audio_state(volume,name)
            if level is not None and before['volume']!=level:hr(method(volume,7,C.c_long,C.c_float,C.c_void_p)(volume,level/100,None))
            if muted is not None and before['muted']!=muted:hr(method(volume,14,C.c_long,W.BOOL,C.c_void_p)(volume,muted,None))
            after=audio_state(volume,name)
        ok=(level is None or abs(after['volume']-level)<=1) and (muted is None or after['muted']==muted)
        return {'ok':ok,'status':'verified' if ok else 'unconfirmed','sound':after,
            'message':'已回读系统声音状态。' if ok else '声音设置尚未确认。'}

    def bluetooth(self,inquiry=False):
        api=C.WinDLL('bthprops.cpl',use_last_error=True)
        definitions={'BluetoothFindFirstRadio':([C.POINTER(RadioParams),C.POINTER(W.HANDLE)],W.HANDLE),
            'BluetoothFindNextRadio':([W.HANDLE,C.POINTER(W.HANDLE)],W.BOOL),'BluetoothFindRadioClose':([W.HANDLE],W.BOOL),
            'BluetoothGetRadioInfo':([W.HANDLE,C.POINTER(RadioInfo)],W.DWORD),
            'BluetoothFindFirstDevice':([C.POINTER(SearchParams),C.POINTER(BluetoothInfo)],W.HANDLE),
            'BluetoothFindNextDevice':([W.HANDLE,C.POINTER(BluetoothInfo)],W.BOOL),'BluetoothFindDeviceClose':([W.HANDLE],W.BOOL)}
        for name,(args,result) in definitions.items():getattr(api,name).argtypes=args;getattr(api,name).restype=result
        kernel=C.WinDLL('kernel32');kernel.CloseHandle.argtypes=[W.HANDLE]
        radio=W.HANDLE();finder=api.BluetoothFindFirstRadio(C.byref(RadioParams(C.sizeof(RadioParams))),C.byref(radio))
        radios=[];devices=[]
        if finder:
            try:
                while True:
                    info=RadioInfo();info.size=C.sizeof(info)
                    if api.BluetoothGetRadioInfo(radio,C.byref(info))==0:radios.append({'name':info.name})
                    kernel.CloseHandle(radio)
                    if not api.BluetoothFindNextRadio(finder,C.byref(radio)):break
            finally:api.BluetoothFindRadioClose(finder)
        search=SearchParams();search.size=C.sizeof(search);search.authenticated=True;search.remembered=True
        search.unknown=bool(inquiry);search.connected=True;search.inquiry=bool(inquiry);search.timeout=2
        device=BluetoothInfo();device.size=C.sizeof(device);found=api.BluetoothFindFirstDevice(C.byref(search),C.byref(device))
        if found:
            try:
                while len(devices)<40:
                    devices.append({'name':device.name,'connected':bool(device.connected),'paired':bool(device.authenticated),
                        'remembered':bool(device.remembered)})
                    device=BluetoothInfo();device.size=C.sizeof(device)
                    if not api.BluetoothFindNextDevice(found,C.byref(device)):break
            finally:api.BluetoothFindDeviceClose(found)
        return {'available':bool(radios),'radios':radios,'devices':devices,'source':'Windows Bluetooth API',
            'controls':['scan'],'pairing_supported':False,'note':'Classic Bluetooth discovery; pairing is not implemented.'}

    def devices(self):
        api=C.WinDLL('setupapi',use_last_error=True)
        api.SetupDiGetClassDevsW.argtypes=[C.c_void_p,W.LPCWSTR,W.HWND,W.DWORD];api.SetupDiGetClassDevsW.restype=W.HANDLE
        api.SetupDiEnumDeviceInfo.argtypes=[W.HANDLE,W.DWORD,C.POINTER(DeviceInfo)];api.SetupDiEnumDeviceInfo.restype=W.BOOL
        api.SetupDiGetDeviceRegistryPropertyW.argtypes=[W.HANDLE,C.POINTER(DeviceInfo),W.DWORD,C.POINTER(W.DWORD),C.c_void_p,W.DWORD,C.POINTER(W.DWORD)]
        api.SetupDiGetDeviceRegistryPropertyW.restype=W.BOOL;api.SetupDiDestroyDeviceInfoList.argtypes=[W.HANDLE]
        handle=api.SetupDiGetClassDevsW(None,None,None,6)
        if handle==C.c_void_p(-1).value:raise HostError('devices_unavailable','无法枚举当前设备。')
        rows=[];classes={}
        def property(info,key):
            value=C.create_unicode_buffer(512);kind=W.DWORD();needed=W.DWORD()
            return value.value if api.SetupDiGetDeviceRegistryPropertyW(handle,C.byref(info),key,C.byref(kind),value,C.sizeof(value),C.byref(needed)) else ''
        try:
            for index in range(512):
                info=DeviceInfo();info.size=C.sizeof(info)
                if not api.SetupDiEnumDeviceInfo(handle,index,C.byref(info)):break
                category=property(info,7);classes[category]=classes.get(category,0)+1
                if category in ('Bluetooth','AudioEndpoint','Display','Monitor','Keyboard','Mouse','Net','USB') and len(rows)<48:
                    name=property(info,12) or property(info,0)
                    rows.append({'name':name,'category':category})
        finally:api.SetupDiDestroyDeviceInfoList(handle)
        return {'available':True,'devices':rows,'class_counts':classes,'source':'Windows SetupAPI','controls':['refresh']}
