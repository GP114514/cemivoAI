"use strict";
(() => {
  const $=id=>document.getElementById(id),image=$("image"),form=$("input"),text=$("text"),send=$("send"),notice=$("notice"),empty=$("empty"),progress=$("progress"),pulse=$("pulse"),activity=$("activity"),stage=$("stage"),elapsed=$("elapsed"),help=$("help"),helpText=$("helpText"),helpLink=$("helpLink");
  const requestKey="cemivoAI.screen.pending",draftKey="cemivoAI.screen.draft",terminal=new Set(["complete","preview","render_failed","partial_failure","needs_input","interrupted"]);
  let frame=null,busy=false,polling=false,statusPolling=false,anchor=null,composing=false,uncertain=null,noticeTimer=null,pulseTimer=null,epoch=0,desktopAttempted=false,operation=null,plannerConfigured=null,noticeKind="",disconnected=false;
  function readStored(key){try{return sessionStorage.getItem(key);}catch{return null;}}
  function remember(key,value){try{if(value===null)sessionStorage.removeItem(key);else sessionStorage.setItem(key,value);}catch{/* Storage is optional; this page never resends a pending request. */}}
  const savedRequest=readStored(requestKey);if(savedRequest&&/^[a-zA-Z0-9_-]{8,80}$/.test(savedRequest))uncertain=savedRequest;
  const draft=readStored(draftKey);if(draft)text.value=draft;
  function put(node,value){if(node.textContent!==value)node.textContent=value;}
  function message(value,persistent=false,kind=""){noticeKind=kind;put(notice,value);notice.hidden=false;clearTimeout(noticeTimer);if(!persistent)noticeTimer=setTimeout(()=>{notice.hidden=true;},5000);}
  function updateHelp(){
    help.hidden=plannerConfigured!==false&&!uncertain&&!disconnected;
    if(plannerConfigured===false){put(helpText,"连接后，即可点击或输入");put(helpLink,"连接 DeepSeek ↗");helpLink.href="/live#plannerKey";}
    else{put(helpText,uncertain?"正在查询上次操作的结果":"正在重新连接");put(helpLink,"打开控制台 ↗");helpLink.href="/live";}
  }
  function seconds(){return operation?Math.max(0,Math.floor((Date.now()-operation.started)/1000)):0;}
  function updateActivity(){activity.hidden=!busy;progress.hidden=!busy;if(!busy||!operation)return;put(stage,operation.label);put(elapsed,seconds()+" 秒");}
  function beginOperation(label,kind){busy=true;operation={started:Date.now(),label,kind};document.body.dataset.busy="true";send.disabled=true;notice.hidden=true;clearTimeout(noticeTimer);updateActivity();}
  function endOperation(){busy=false;operation=null;document.body.dataset.busy="false";send.disabled=false;updateActivity();}
  function resultMessage(value,persistent=false){message(value+" · "+seconds()+" 秒",persistent);}
  function clickFeedback(event){clearTimeout(pulseTimer);pulse.style.left=event.clientX+"px";pulse.style.top=event.clientY+"px";pulse.style.animation="none";pulse.hidden=false;void pulse.offsetWidth;pulse.style.animation="";pulseTimer=setTimeout(()=>{pulse.hidden=true;},600);}
  async function api(path,data){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),data?240000:8000);try{const response=await fetch(path,{method:data?"POST":"GET",cache:"no-store",headers:{"Content-Type":"application/json"},body:data?JSON.stringify(data):undefined,signal:controller.signal});const result=await response.json();if(!response.ok){const error=new Error(result.error||"请求未完成");error.code=result.code;error.status=response.status;throw error;}return result;}finally{clearTimeout(timer);}}
  function point(event){if(!frame)return null;const rect=image.getBoundingClientRect();if(!rect.width||!rect.height)return null;const scale=Math.min(rect.width/frame.width,rect.height/frame.height),left=rect.left+(rect.width-frame.width*scale)/2,top=rect.top+(rect.height-frame.height*scale)/2;const x=(event.clientX-left)/scale,y=(event.clientY-top)/scale;return x>=0&&y>=0&&x<frame.width&&y<frame.height?{x:Math.floor(x),y:Math.floor(y)}:null;}
  function validFrame(next){return next&&typeof next.session_id==="string"&&Number.isInteger(next.version)&&next.version>=1&&Number.isInteger(next.width)&&Number.isInteger(next.height)&&next.width>0&&next.height>0&&next.width<=1024&&next.height<=1024&&typeof next.sha256==="string"&&/^[a-f0-9]{64}$/i.test(next.sha256)&&typeof next.image_url==="string";}
  async function display(next,expectedEpoch=epoch){
    if(expectedEpoch!==epoch)return false;
    if(!validFrame(next))throw new Error("模型画面回执不完整，已有图片已保留。");
    if(frame&&frame.session_id===next.session_id&&next.version<frame.version)return false;
    const source=new URL(next.image_url,location.origin);
    if(source.origin!==location.origin||!/^\/api\/live\/images\/[a-f0-9]+\.png$/.test(source.pathname)||source.search||source.hash)throw new Error("画面地址不正确");
    const firstFrame=!frame,shouldDecode=!frame||next.sha256!==frame.sha256||image.hidden;
    if(shouldDecode){const decoded=new Image();decoded.src=source.href;let timer;try{await Promise.race([decoded.decode(),new Promise((resolve,reject)=>{timer=setTimeout(()=>reject(new Error("新画面暂时无法读取，当前画面已保留。")),8000);})]);}finally{clearTimeout(timer);}if(decoded.naturalWidth!==next.width||decoded.naturalHeight!==next.height)throw new Error("画面尺寸不匹配");}
    // An earlier polling decode cannot overwrite a deliberate action's result.
    if(expectedEpoch!==epoch)return false;
    if(frame&&frame.session_id===next.session_id&&next.version<frame.version)return false;
    if(shouldDecode)image.src=source.href;
    if(anchor&&(anchor.session_id!==next.session_id||anchor.version!==next.version||anchor.sha256!==next.sha256||anchor.image_url!==next.image_url)){form.hidden=true;anchor=null;}
    frame=next;image.hidden=false;empty.hidden=true;document.body.dataset.preview=String(next.edit?.preview===true);
    if(next.edit?.preview)message(next.agent?.warning||"待确认预览：内容未通过核验，原输入文字已保留。",true);
    else if(firstFrame&&!busy&&!uncertain)message("左键操作，右键输入文字");
    return true;
  }
  async function pollStatus(){
    if(statusPolling)return;statusPolling=true;const stamp=epoch;
    try{const status=await api("/api/live/status");if(typeof status.planner?.configured==="boolean")plannerConfigured=status.planner.configured;updateHelp();
      if(busy&&operation&&stamp===epoch){const labels=operation.kind==="desktop"?{understanding:"正在整理首屏",refining:"正在检查首屏",drawing:"正在绘制首屏",verifying:"正在检查生成结果"}:{understanding:"正在理解你的操作",refining:"正在确认点击细节",acting:"正在完成你的任务",drawing:"正在绘制新画面",verifying:"正在检查生成结果"};operation.label=status.runtime?.busy===true?(labels[status.runtime.stage]||"正在处理，请稍候"):"正在等待操作结果";updateActivity();}
    }catch{if(busy&&operation&&stamp===epoch){operation.label="连接暂时中断，正在等待结果";updateActivity();}}
    finally{statusPolling=false;}
  }
  async function createDesktop(){
    if(desktopAttempted||frame||busy||uncertain)return;
    desktopAttempted=true;const stamp=++epoch;beginOperation("正在生成你的首屏","desktop");empty.textContent="正在准备你的画面…";empty.hidden=false;
    try{const next=await api("/api/live/desktop",{});await display(next,stamp);if(!next.edit?.preview)resultMessage("画面已就绪，右键输入文字");}
    catch(error){if(error.code==="planner_not_configured")plannerConfigured=false;empty.textContent="画面尚未就绪，可到控制台继续。";resultMessage(error.message||"首屏生成未完成；不会自动重复生成。",true);updateHelp();}
    finally{endOperation();}
  }
  async function sync(){
    updateActivity();if(busy){void pollStatus();return;}if(polling)return;polling=true;const stamp=epoch;
    try{
      if(uncertain){try{const saved=await api("/api/agent/request?id="+encodeURIComponent(uncertain));if(terminal.has(saved.status)){uncertain=null;remember(requestKey,null);message(saved.warning||saved.answer||"执行结果已返回");updateHelp();}}catch{message("执行结果仍待确认，正在继续查询；不会自动重复发送。",true);}}
      let next;
      try{next=await api("/api/live/current");}
      catch(error){if(!frame&&!desktopAttempted&&!uncertain&&(error.status===404||["session_missing","invalid_session","session_expired"].includes(error.code))){await createDesktop();return;}throw error;}
      if(stamp!==epoch||busy)return;await display(next,stamp);if(stamp!==epoch||busy)return;
      if(disconnected&&noticeKind==="connection"){message("连接已恢复");}disconnected=false;updateHelp();
    }catch(error){if(stamp!==epoch||busy)return;disconnected=true;updateHelp();if(!frame){empty.hidden=false;empty.textContent=desktopAttempted?"等待已生成的模型画面…":"暂时无法连接，正在重试…";}if(frame&&error.message)message("连接暂时中断，当前画面保留；正在重试。",true,"connection");}
    finally{polling=false;void pollStatus();}
  }
  async function act(event,source=frame,feedback=null){
    if(busy){message("上一操作处理中，本次没有重复发送或排队。",true);return false;}
    if(!source){message("正在等待模型画面，请稍候。");return false;}
    if(uncertain){message("上次任务结果待确认，正在继续查询；这次没有重复发送。",true);updateHelp();return false;}
    if(frame&&(source.session_id!==frame.session_id||source.version!==frame.version||source.sha256!==frame.sha256||source.image_url!==frame.image_url)){message("画面已变化，文字已保留，请重新右键选择位置。");return false;}
    const id=crypto.randomUUID(),stamp=++epoch;beginOperation("已收到，正在理解你的操作","action");form.hidden=true;anchor=null;if(feedback)clickFeedback(feedback);remember(draftKey,text.value);remember(requestKey,id);let received=false;
    try{const result=await api("/api/agent/event",{session_id:source.session_id,version:source.version,request_id:id,interaction_mode:"auto",event});received=true;remember(requestKey,null);await display(result,stamp);const status=result.agent?.status;
      if(["render_failed","partial_failure","interrupted","needs_input"].includes(status)){resultMessage(result.agent.warning||result.agent.answer||"本次没有完成，原文字已保留。",true);return false;}
      if(status==="preview"){resultMessage(result.agent.warning||"已显示待确认预览，文字已保留。",true);return true;}
      resultMessage(result.sha256===source.sha256?(result.state?.notice||result.plan?.intent||"当前画面保持不变"):(result.plan?.expected_change||"画面已更新"));return true;
    }catch(error){const definite=["busy","stale_frame","frame_expired","session_missing","invalid_session","session_expired","invalid_event","invalid_request","planner_not_configured","invalid_config"].includes(error.code);
      if(!received&&!definite){uncertain=id;resultMessage("执行结果暂未确认，正在查询；不会重复发送。",true);}
      else{remember(requestKey,null);if(error.code==="planner_not_configured")plannerConfigured=false;resultMessage(error.message||"操作未完成，文字已保留。",true);}updateHelp();return false;
    }finally{endOperation();void sync();}
  }
  image.addEventListener("click",event=>{if(!form.hidden){form.hidden=true;anchor=null;message("已关闭输入框，文字保留；请再点击要操作的位置。");return;}const position=point(event);if(position)void act({type:"click",...position,input_method:"pointer"},frame,event);});
  image.addEventListener("contextmenu",event=>{event.preventDefault();if(busy){message("上一操作处理中，请等待。");return;}const position=point(event);if(!position)return;anchor={...frame,...position};form.hidden=false;const rect=form.getBoundingClientRect();form.style.left=Math.max(8,Math.min(event.clientX,innerWidth-Math.min(320,innerWidth-16)-8))+"px";form.style.top=Math.max(8,Math.min(event.clientY,innerHeight-rect.height-8))+"px";text.focus();});
  form.addEventListener("submit",async event=>{event.preventDefault();if(composing||busy||!anchor)return;const value=text.value;if(!value.trim()){text.focus();return;}if(value.length>500||new TextEncoder().encode(value).length>1500){message("文字过长，已保留原文。");return;}const target=anchor;await act({type:"click",x:target.x,y:target.y,text:value,input_method:"contextmenu"},target);});
  text.addEventListener("input",()=>{remember(draftKey,text.value);});text.addEventListener("compositionstart",()=>{composing=true;});text.addEventListener("compositionend",()=>{composing=false;remember(draftKey,text.value);});text.addEventListener("keydown",event=>{if(event.key==="Escape"){form.hidden=true;anchor=null;}if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing&&!composing&&event.keyCode!==229){event.preventDefault();if(!event.repeat)form.requestSubmit();}});
  window.addEventListener("keydown",event=>{if(event.key==="Escape"){form.hidden=true;anchor=null;}});window.addEventListener("online",()=>{void sync();});window.addEventListener("focus",()=>{void sync();});
  if(uncertain){message("正在查询上次操作的结果，不会重复发送。",true);updateHelp();}
  void sync();setInterval(sync,1000);
})();
