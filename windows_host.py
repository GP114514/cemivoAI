"""Narrow Windows system APIs. No shell, credential access or arbitrary commands."""
from contextlib import contextmanager
import ctypes as C
from ctypes import wintypes as W
import hashlib
import os
import platform
import shutil
import socket
import sys
import time
import uuid


class HostError(RuntimeError):
    def __init__(self,code,message,native_code=None):
        super().__init__(message);self.code=code;self.native_code=native_code


class GUID(C.Structure):
    _fields_=[('Data1',W.DWORD),('Data2',W.WORD),('Data3',W.WORD),('Data4',C.c_ubyte*8)]

class Interface(C.Structure):
    _fields_=[('guid',GUID),('description',W.WCHAR*256),('state',C.c_int)]

class SSID(C.Structure):
    _fields_=[('length',W.ULONG),('data',C.c_ubyte*32)]

class Association(C.Structure):
    _fields_=[('ssid',SSID),('bss_type',C.c_int),('bssid',C.c_ubyte*6),('phy_type',C.c_int),
        ('phy_index',W.ULONG),('signal',W.ULONG),('rx_rate',W.ULONG),('tx_rate',W.ULONG)]

class Security(C.Structure):
    _fields_=[('enabled',W.BOOL),('one_x',W.BOOL),('auth',C.c_int),('cipher',C.c_int)]

class Connection(C.Structure):
    _fields_=[('state',C.c_int),('mode',C.c_int),('profile',W.WCHAR*256),('association',Association),('security',Security)]

class Profile(C.Structure):
    _fields_=[('name',W.WCHAR*256),('flags',W.DWORD)]

class Network(C.Structure):
    _fields_=[('profile',W.WCHAR*256),('ssid',SSID),('bss_type',C.c_int),('bssid_count',W.ULONG),
        ('connectable',W.BOOL),('reason',W.DWORD),('phy_count',W.ULONG),('phys',C.c_int*8),
        ('more_phys',W.BOOL),('signal',W.ULONG),('secured',W.BOOL),('auth',C.c_int),('cipher',C.c_int),
        ('flags',W.DWORD),('reserved',W.DWORD)]

class ConnectParameters(C.Structure):
    _fields_=[('mode',C.c_int),('profile',W.LPCWSTR),('ssid',C.c_void_p),('bssids',C.c_void_p),
        ('bss_type',C.c_int),('flags',W.DWORD)]

class MemoryStatus(C.Structure):
    _fields_=[('length',W.DWORD),('load',W.DWORD)]+[(name,C.c_ulonglong) for name in
        ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended')]


STATES={0:'not_ready',1:'connected',2:'ad_hoc',3:'disconnecting',4:'disconnected',5:'associating',6:'discovering',7:'authenticating'}

def profile_id(guid,name):
    return hashlib.sha256((guid+'\0'+name).encode('utf-8')).hexdigest()[:24]

def _ssid(value):return bytes(value.data[:min(value.length,32)]).decode('utf-8',errors='replace')

def _check(code,operation):
    if code:
        message=('Windows 拒绝访问；请在系统设置中检查 Wi-Fi/位置权限。' if code==5 else
            'Windows WLAN 服务不可用，请检查无线网卡和 WLAN 服务。' if code==1062 else
            'Windows 未完成 '+operation+'。')
        raise HostError('windows_access_denied' if code==5 else 'windows_api_error',message,int(code))


class WindowsHost:
    def __init__(self):
        self.available=sys.platform=='win32'
        self.wlan=None

    def _api(self):
        if not self.available:raise HostError('unsupported_platform','本机系统连接目前仅支持 Windows。')
        if self.wlan is None:
            api=C.WinDLL('wlanapi',use_last_error=True)
            definitions={
                'WlanOpenHandle':([W.DWORD,C.c_void_p,C.POINTER(W.DWORD),C.POINTER(W.HANDLE)],W.DWORD),
                'WlanCloseHandle':([W.HANDLE,C.c_void_p],W.DWORD),
                'WlanFreeMemory':([C.c_void_p],None),
                'WlanEnumInterfaces':([W.HANDLE,C.c_void_p,C.POINTER(C.c_void_p)],W.DWORD),
                'WlanGetProfileList':([W.HANDLE,C.POINTER(GUID),C.c_void_p,C.POINTER(C.c_void_p)],W.DWORD),
                'WlanGetAvailableNetworkList':([W.HANDLE,C.POINTER(GUID),W.DWORD,C.c_void_p,C.POINTER(C.c_void_p)],W.DWORD),
                'WlanQueryInterface':([W.HANDLE,C.POINTER(GUID),C.c_int,C.c_void_p,C.POINTER(W.DWORD),C.POINTER(C.c_void_p),C.POINTER(C.c_int)],W.DWORD),
                'WlanConnect':([W.HANDLE,C.POINTER(GUID),C.POINTER(ConnectParameters),C.c_void_p],W.DWORD),
                'WlanScan':([W.HANDLE,C.POINTER(GUID),C.c_void_p,C.c_void_p,C.c_void_p],W.DWORD)}
            for name,(args,result) in definitions.items():
                function=getattr(api,name);function.argtypes=args;function.restype=result
            self.wlan=api
        return self.wlan

    @contextmanager
    def handle(self):
        api=self._api();version=W.DWORD();handle=W.HANDLE()
        _check(api.WlanOpenHandle(2,None,C.byref(version),C.byref(handle)),'打开 WLAN')
        try:yield api,handle
        finally:api.WlanCloseHandle(handle,None)

    @staticmethod
    def _list(api,call,item_type):
        data=C.c_void_p();_check(call(C.byref(data)),'读取 WLAN 信息')
        try:
            count=W.DWORD.from_address(data.value).value
            if count>1024:raise HostError('invalid_native_data','Windows 返回的列表过大。')
            return [item_type.from_buffer_copy(C.string_at(data.value+8+i*C.sizeof(item_type),C.sizeof(item_type))) for i in range(count)]
        finally:api.WlanFreeMemory(data)

    @staticmethod
    def _query(api,handle,guid,opcode,item_type):
        size=W.DWORD();data=C.c_void_p();kind=C.c_int()
        _check(api.WlanQueryInterface(handle,C.byref(guid),opcode,None,C.byref(size),C.byref(data),C.byref(kind)),'读取连接状态')
        try:
            if size.value<C.sizeof(item_type):raise HostError('invalid_native_data','Windows 返回的连接数据不完整。')
            return item_type.from_buffer_copy(C.string_at(data,C.sizeof(item_type)))
        finally:api.WlanFreeMemory(data)

    def wifi(self,scan=False):
        result={'available':False,'interfaces':[],'networks':[],'profiles':[],'errors':[],'source':'Windows WLAN API'}
        try:
            with self.handle() as (api,handle):
                interfaces=self._list(api,lambda p:api.WlanEnumInterfaces(handle,None,p),Interface)
                for interface in interfaces:
                    guid=str(uuid.UUID(bytes_le=bytes(interface.guid)));connection=None
                    row={'id':guid,'description':interface.description,'state':STATES.get(interface.state,'unknown')}
                    if interface.state==1:
                        try:
                            connection=self._query(api,handle,interface.guid,7,Connection)
                            row.update(profile_id=profile_id(guid,connection.profile),ssid=_ssid(connection.association.ssid),
                                profile=connection.profile,signal_percent=int(connection.association.signal),
                                rx_mbps=round(connection.association.rx_rate/1000,1),tx_mbps=round(connection.association.tx_rate/1000,1),
                                secured=bool(connection.security.enabled))
                        except HostError as error:result['errors'].append(self.error(error))
                    result['interfaces'].append(row)
                    try:
                        profiles=self._list(api,lambda p:api.WlanGetProfileList(handle,C.byref(interface.guid),None,p),Profile)
                        result['profiles'].extend({'id':profile_id(guid,p.name),'name':p.name,'interface_id':guid,
                            'connected':bool(connection and connection.profile==p.name)} for p in profiles)
                    except HostError as error:result['errors'].append(self.error(error))
                    if scan:
                        code=api.WlanScan(handle,C.byref(interface.guid),None,None,None)
                        if code:result['errors'].append(self.error(HostError('scan_not_started','Windows 未启动刷新扫描。',code)))
                    try:
                        networks=self._list(api,lambda p:api.WlanGetAvailableNetworkList(handle,C.byref(interface.guid),0,None,p),Network)
                        result['networks'].extend({'ssid':_ssid(n.ssid),'signal_percent':int(n.signal),'secured':bool(n.secured),
                            'connectable':bool(n.connectable),'connected':bool(n.flags&1),'interface_id':guid,
                            'profile_id':profile_id(guid,n.profile) if n.profile else None} for n in networks[:40])
                    except HostError as error:result['errors'].append(self.error(error))
                result['available']=bool(interfaces)
                result['scan_requested']=bool(scan)
        except (HostError,OSError) as error:result['errors'].append(self.error(error))
        return result

    @staticmethod
    def error(error):return {'code':getattr(error,'code','windows_unavailable'),'message':str(error),
        'native_code':getattr(error,'native_code',None)}

    def connect(self,profile):
        before=self.wifi();saved=next((p for p in before['profiles'] if p['id']==profile),None)
        if saved is None:raise HostError('profile_not_found','该网络不在 Windows 已保存配置中，请刷新后重试。')
        current=next((i for i in before['interfaces'] if i.get('profile_id')==profile and i['state']=='connected'),None)
        if current:
            return {'ok':True,'status':'already_connected','message':'Windows 已连接该网络，未重复重连。','connection':current}
        visible=any(n['profile_id']==profile and n['connectable'] for n in before['networks'])
        if not visible:raise HostError('network_not_visible','已保存网络当前不可见或不可连接，请靠近网络后刷新。')
        with self.handle() as (api,handle):
            guid=GUID.from_buffer_copy(uuid.UUID(saved['interface_id']).bytes_le)
            parameters=ConnectParameters(0,saved['name'],None,None,1,0)
            _check(api.WlanConnect(handle,C.byref(guid),C.byref(parameters),None),'连接 Wi-Fi')
        # WlanConnect acknowledges admission, not completion. Read actual state.
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            after=self.wifi()
            current=next((i for i in after['interfaces'] if i.get('profile_id')==profile and i['state']=='connected'),None)
            if current:return {'ok':True,'status':'connected','message':'Windows 已确认连接。','connection':current}
            time.sleep(.4)
        return {'ok':False,'status':'unconfirmed','error':{'code':'connection_unconfirmed',
            'message':'Windows 已接收连接请求，但尚未确认连接；请刷新真实状态，不会自动重试。'}}

    def info(self):
        result={'connected':self.available,'source':'local Windows APIs','name':socket.gethostname(),
            'os':platform.system(),'version':platform.version(),'architecture':platform.machine(),'cpu_threads':os.cpu_count()}
        if not self.available:return result
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                result['cpu']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
            memory=MemoryStatus();memory.length=C.sizeof(memory)
            kernel=C.WinDLL('kernel32',use_last_error=True)
            kernel.GlobalMemoryStatusEx.argtypes=[C.POINTER(MemoryStatus)];kernel.GlobalMemoryStatusEx.restype=W.BOOL
            if kernel.GlobalMemoryStatusEx(C.byref(memory)):
                result.update(ram_gb=round(memory.total_physical/1024**3,1),ram_available_gb=round(memory.available_physical/1024**3,1))
            disk=shutil.disk_usage(os.environ.get('SystemDrive','C:')+'\\')
            result['system_disk']={'total_gb':round(disk.total/1024**3,1),'free_gb':round(disk.free/1024**3,1)}
        except OSError as error:result['error']=self.error(error)
        return result

    def open_settings(self,page):
        pages={'wifi':'network-wifi','network':'network-status','bluetooth':'bluetooth',
            'devices':'connecteddevices','about':'about','display':'display','sound':'sound'}
        if page not in pages:raise HostError('unsupported_setting','不支持的系统设置页面。')
        if not self.available:raise HostError('unsupported_platform','系统设置仅支持本地 Windows。')
        shell=C.WinDLL('shell32',use_last_error=True)
        shell.ShellExecuteW.argtypes=[W.HWND,W.LPCWSTR,W.LPCWSTR,W.LPCWSTR,W.LPCWSTR,C.c_int]
        shell.ShellExecuteW.restype=C.c_void_p
        code=shell.ShellExecuteW(None,'open','ms-settings:'+pages[page],None,None,1)
        if not code or code<=32:raise HostError('settings_launch_failed','Windows 未接受打开设置的请求。',code)
        return {'ok':True,'status':'settings_requested','page':page,
            'message':'已请求 Windows 打开系统设置；这不代表网络已连接或设备已配对。'}
