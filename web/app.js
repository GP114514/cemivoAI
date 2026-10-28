(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const form=$('generateForm'),prompt=$('prompt'),preview=$('preview'),seed=$('seed'),generateButton=$('generateButton');
  const states={preparing:'准备训练环境',running:'神经网络训练中',paused:'训练已暂停',complete:'训练已完成',failed:'训练出现错误'};
  let history=[],current=null,ready=false,connected=false,busy=false,statusBusy=false;
  let controller=null,requestNumber=0,generationCount=0;
  let model={},training={},displayedTrainingUrl=null,displayedTrainingStep=null;
  const number=value=>value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value))?Number(value):null;
  const format=value=>number(value)===null?'—':Number(value).toLocaleString('zh-CN');
  function setMessage(text='',error=false,retry=false){
    const box=$('message');box.replaceChildren();box.hidden=!text;box.className=`message${error?' error':''}`;
    if(!text)return;box.append(document.createTextNode(text));
    if(retry){const button=document.createElement('button');button.type='button';button.textContent='重新连接';button.addEventListener('click',()=>loadStatus());box.append(button);}
  }
  function syncControls(){
    generateButton.disabled=busy||!ready||!connected;$('variantButton').disabled=busy||!ready||!connected||!current;
    $('randomSeed').disabled=busy;$('steps').disabled=busy||!connected;$('busyIndicator').hidden=!busy;$('canvasArea').classList.toggle('working',busy);
    generateButton.querySelector('.generate-label').textContent=busy?'正在神经网络采样…':!connected?'等待连接模型':!ready?'等待模型就绪':'生成像素图像';form.setAttribute('aria-busy',String(busy));
  }
  function updatePromptState(){
    const bytes=new TextEncoder().encode(prompt.value).length;$('characterCount').textContent=`${bytes} / 190 字节`;document.querySelectorAll('.example-chip').forEach(chip=>chip.classList.toggle('active',chip.dataset.example===prompt.value));
  }
  function setTiming(value,title=''){
    const actual=number(value);$('resultTiming').replaceChildren(document.createTextNode(actual===null?'— ':`${actual<100?actual.toFixed(1):Math.round(actual)} `));const unit=document.createElement('small');unit.textContent='ms';$('resultTiming').append(unit);$('resultTiming').title=title;
  }
  function imageUrl(value){return typeof value==='string'&&/^\/api\/images\/[a-zA-Z0-9_-]+\.png$/.test(value)?value:null;}
  function previewUrl(value){return value==='/api/training-preview.png'?value:null;}
  function setDownload(url,item){
    const link=$('downloadPng');link.classList.toggle('disabled',!url);link.setAttribute('aria-disabled',String(!url));if(url){link.href=url;link.download=`cemivo-neural-step${item.model?.checkpoint_step??'unknown'}-seed${item.seed}.png`;}else link.removeAttribute('href');
  }
  function showResult(item,restore=false){
    current=item;preview.src=item.objectUrl;preview.alt=`真实神经网络采样：${item.prompt}`;preview.hidden=false;$('emptyState').hidden=true;
    $('previewSource').textContent=`用户采样 · ${item.steps} 步 · 种子 ${item.seed}`;$('resultLabel').textContent='神经网络采样';$('resultNumber').textContent=`/ ${String(item.number).padStart(3,'0')}`;
    $('resultDimensions').textContent=`${item.width} × ${item.height}`;$('resultCheckpoint').textContent=`检查点 ${format(item.model?.checkpoint_step)} 步`;
    setTiming(item.timings?.sample_ms,`采样 ${number(item.timings?.sample_ms)??'—'} ms · PNG 编码 ${number(item.timings?.encode_ms)??'—'} ms · 服务总耗时 ${number(item.timings?.total_ms)??'—'} ms · 请求和图片下载 ${item.request_ms.toFixed(1)} ms`);
    $('resultIndicator').classList.add('ready');$('fullscreenButton').disabled=false;setDownload(imageUrl(item.png_url),item);
    if(restore){prompt.value=item.prompt;seed.value=item.seed;if([...$('steps').options].some(option=>Number(option.value)===item.steps))$('steps').value=String(item.steps);updatePromptState();}
    renderHistory();syncControls();
  }
  function renderHistory(){
    $('history').replaceChildren();$('historyCount').textContent=String(history.length);
    history.forEach(item=>{const button=document.createElement('button');button.type='button';button.className=`history-item${current?.id===item.id?' active':''}`;
      button.title=`${item.prompt} · 采样 ${item.steps} 步 · 检查点 ${item.model?.checkpoint_step??'未知'}`;button.setAttribute('aria-label',`查看第 ${item.number} 张真实采样`);button.setAttribute('aria-pressed',String(current?.id===item.id));
      const image=document.createElement('img');image.src=item.objectUrl;image.alt='';image.loading='lazy';const label=document.createElement('span');label.textContent=`${String(item.number).padStart(3,'0')} · ${item.steps}步`;
      button.append(image,label);button.addEventListener('click',()=>{if(busy)return;setMessage();showResult(item,true);});$('history').append(button);
    });
  }
  function updateTrainingPreview(status){
    const path=previewUrl(status.preview_url);if(!path||current||busy)return;
    const previewStep=status.preview_step??status.training?.preview_step,version=previewStep??status.training?.updated_at??status.model?.checkpoint_step??Date.now();
    const url=`${path}?v=${encodeURIComponent(version)}`;if(url===displayedTrainingUrl)return;displayedTrainingUrl=url;displayedTrainingStep=number(previewStep);
    preview.src=url;preview.alt='最近保存的真实神经网络训练预览，早期可能仍为噪声';preview.hidden=false;$('emptyState').hidden=true;
    $('previewSource').textContent=displayedTrainingStep===null?'训练预览 · 最近保存的模型输出':`训练预览 · 实际第 ${format(displayedTrainingStep)} 步`;
    $('resultLabel').textContent='真实训练预览';$('resultCheckpoint').textContent=displayedTrainingStep===null?'图片步数未提供':`预览 ${format(displayedTrainingStep)} 步`;
    $('resultNumber').textContent='/ TRAIN';setTiming(null);setDownload(null);$('fullscreenButton').disabled=false;
  }
  function elapsed(value){const seconds=number(value);if(seconds===null)return'—';if(seconds<60)return`${Math.floor(seconds)} 秒`;if(seconds<3600)return`${Math.floor(seconds/60)} 分 ${Math.floor(seconds%60)} 秒`;return`${Math.floor(seconds/3600)} 时 ${Math.floor(seconds%3600/60)} 分`;}
  function updated(value){if(value===null||value===undefined)return'每 5 秒更新';const date=typeof value==='number'?new Date(value<1e12?value*1000:value):new Date(value);return Number.isNaN(date.valueOf())?'每 5 秒更新':`更新于 ${date.toLocaleTimeString('zh-CN',{hour12:false})}`;}
  function displayStatus(status){
    model=status.model||{};training=status.training||{};const params=number(model.parameters),step=number(training.step),total=number(training.total_steps);
    $('connectionBadge').textContent='本地已连接';$('trainingState').textContent=states[training.state]||'读取训练状态';$('trainingUpdated').textContent=updated(training.updated_at);
    $('trainingModel').textContent=`${model.name||'神经像素模型'} · ${params===null?'参数待确认':`${params>=1e8?(params/1e8).toFixed(2)+' 亿':format(params)} 参数`}`;$('trainingDot').className=`training-dot ${training.state||''}`;
    $('trainingStep').textContent=step===null?'—':`${format(step)}${total!==null&&total>0?' / '+format(total):''}`;const loss=number(training.loss);$('trainingLoss').textContent=loss===null?'—':loss.toFixed(6);
    $('trainingSamples').textContent=format(training.samples_seen);$('trainingElapsed').textContent=elapsed(training.elapsed_seconds);
    const hasProgress=step!==null&&total!==null&&total>0;$('trainingProgress').hidden=!hasProgress;
    if(hasProgress){const progress=Math.max(0,Math.min(100,step/total*100));$('trainingProgressFill').style.width=`${progress}%`;$('trainingProgress').setAttribute('aria-valuenow',String(step));$('trainingProgress').setAttribute('aria-valuemin','0');$('trainingProgress').setAttribute('aria-valuemax',String(total));$('trainingProgress').setAttribute('aria-valuetext',`已完成 ${format(step)} / ${format(total)} 步`);}
    const descriptions={preparing:'正在准备数据与 GPU 运行环境。预览仅在模型实际输出后显示。',running:'训练正在进行。损失与步数是真实进度；早期输出可能仍是噪声，文字与界面细节尚需训练。',paused:'训练已暂停。若存在可用检查点，可以用它采样；结果代表该检查点的实际能力。',complete:'训练任务已完成。完成步数不等于掌握提示词、文字或界面细节，请以实际采样结果评估。',failed:'训练发生错误。下方显示本机服务返回的实际错误。'};
    $('trainingDescription').textContent=descriptions[training.state]||'训练进度和损失来自本机服务，不代表图像质量承诺。';$('trainingError').hidden=!training.error;$('trainingError').textContent=typeof training.error==='string'?training.error:'';
    $('modelParameters').textContent=format(params);$('modelCheckpoint').textContent=format(model.checkpoint_step);
    $('modelDescription').textContent=`${model.name||'当前模型'}${params===null?'':`包含 ${format(params)} 个参数${params>=1e8?'，属于亿级参数像素生成模型':''}`}，直接学习从提示词与噪声生成像素图像。实际能力以当前检查点的采样结果为准。`;
    $('initializationStatus').textContent=model.random_initialization===true?'本机服务确认：模型全部从随机权重初始化。':'本机服务尚未确认全部随机初始化状态。';
    const width=number(model.native_width)||256,height=number(model.native_height)||256;$('nativeDimensions').textContent=`${width} × ${height}`;if(!current&&!displayedTrainingUrl)$('resultDimensions').textContent=`${width} × ${height}`;
    $('engineStatus').textContent=`${model.name||'像素模型'} · 检查点 ${format(model.checkpoint_step)} 步`;
    const allowed=Array.isArray(status.capabilities?.sampling_steps)?status.capabilities.sampling_steps.filter(value=>Number.isInteger(value)&&value>0&&value<=256):[];
    if(allowed.length){const previous=Number($('steps').value),signature=allowed.join(',');if($('steps').dataset.allowed!==signature){$('steps').replaceChildren(...allowed.map(value=>{const option=document.createElement('option');option.value=String(value);option.textContent=`${value} 步${value===4?' · 较少计算':value===32?' · 较多计算':''}`;return option;}));$('steps').value=String(allowed.includes(previous)?previous:allowed[0]);$('steps').dataset.allowed=signature;}}
    if(!current&&!status.preview_url){$('emptyState').querySelector('p').textContent=ready?'模型已就绪，等待首次采样':'等待实际模型输出';$('emptyState').querySelector('span').textContent='不使用模板或示意图片填充预览';}
    updateTrainingPreview(status);syncControls();
  }
  async function loadStatus(){
    if(statusBusy)return;statusBusy=true;
    try{const response=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(8000)});if(!response.ok)throw new Error('无法读取本机训练状态');const status=await response.json();if(status.app_id!=='cemivoAI.neural')throw new Error('当前连接的不是神经像素生成服务，请重启新版服务。');
      connected=true;ready=status.ready===true;displayStatus(status);if($('message').dataset.connectionError==='true'){setMessage();$('message').dataset.connectionError='false';}
    }catch(error){connected=false;ready=false;syncControls();$('connectionBadge').textContent='服务未连接';$('trainingState').textContent='训练状态暂不可用';$('engineStatus').textContent='本机服务连接失败';if(!busy){setMessage(error.message||'无法连接本机神经网络服务，请检查启动程序。',true,true);$('message').dataset.connectionError='true';}}
    finally{statusBusy=false;}
  }
  async function generate(isVariant=false){
    if(busy)return;if(!ready||!connected){setMessage('真实模型检查点尚未就绪，请等待训练状态更新。',true);return;}
    const text=prompt.value.trim();if(!text){setMessage('先写一句画面描述。',true);prompt.focus();return;}const promptBytes=new TextEncoder().encode(text).length;if(promptBytes>190){setMessage(`这段描述有 ${promptBytes} 个 UTF-8 字节，模型最多支持 190 个。请精简描述，建议60个汉字以内；英文也按实际字节计算。`,true);prompt.focus();return;}let nextSeed=Number(seed.value);if(!Number.isInteger(nextSeed)||nextSeed<0||nextSeed>2147483647){setMessage('随机种子需为 0–2147483647 的整数。',true);return;}
    if(isVariant){nextSeed=(nextSeed+1)%2147483648;seed.value=nextSeed;}const steps=Number($('steps').value),width=number(model.native_width)||256,height=number(model.native_height)||256;
    controller?.abort();controller=new AbortController();const thisController=controller,thisRequest=++requestNumber,abortTimer=setTimeout(()=>thisController.abort(),90000),started=performance.now();busy=true;syncControls();setMessage();
    try{const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:text,seed:nextSeed,steps,width,height}),signal:thisController.signal});let result;try{result=await response.json();}catch{throw new Error('采样服务未返回有效数据。');}if(!response.ok)throw new Error(result.error||result.message||`采样失败（${response.status}）`);
      if(thisRequest!==requestNumber)return;const url=imageUrl(result.png_url);if(!url)throw new Error('采样服务未返回有效的本地图像地址。');const imageResponse=await fetch(url,{signal:thisController.signal,cache:'no-store'});if(!imageResponse.ok)throw new Error('模型采样完成，但读取 PNG 失败。');const blob=await imageResponse.blob();if(!blob.type.startsWith('image/png'))throw new Error('采样结果不是有效的 PNG 图像。');if(thisRequest!==requestNumber)return;
      const item={...result,prompt:text,seed:nextSeed,steps,width:result.width||width,height:result.height||height,number:++generationCount,request_ms:performance.now()-started,objectUrl:URL.createObjectURL(blob)};history.unshift(item);while(history.length>8)URL.revokeObjectURL(history.pop().objectUrl);showResult(item);
    }catch(error){if(thisRequest===requestNumber)setMessage(error.name==='AbortError'?'采样等待超过 90 秒，本次请求已停止等待。模型较大时请检查本机服务进度后重试。':error.message||'神经网络采样失败。',true);}
    finally{clearTimeout(abortTimer);if(thisRequest===requestNumber){busy=false;syncControls();}}
  }
  form.addEventListener('submit',event=>{event.preventDefault();void generate();});prompt.addEventListener('input',updatePromptState);$('variantButton').addEventListener('click',()=>void generate(true));
  $('randomSeed').addEventListener('click',()=>{const values=new Uint32Array(1);crypto.getRandomValues(values);seed.value=values[0]%2147483648;});document.querySelectorAll('.example-chip').forEach(chip=>chip.addEventListener('click',()=>{prompt.value=chip.dataset.example;updatePromptState();prompt.focus();}));
  document.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'&&!$('aboutDialog').open&&!$('previewDialog').open){event.preventDefault();void generate();}});
  $('aboutButton').addEventListener('click',()=>$('aboutDialog').showModal());$('closeAbout').addEventListener('click',()=>$('aboutDialog').close());
  $('fullscreenButton').addEventListener('click',()=>{const source=current?.objectUrl||displayedTrainingUrl;if(!source)return;$('largePreview').src=source;$('fullscreenTitle').textContent=current?`真实采样 · ${current.steps} 步 · 检查点 ${format(current.model?.checkpoint_step)} 步`:'最近保存的真实训练预览 · 早期可能为噪声';$('previewDialog').showModal();});$('closePreview').addEventListener('click',()=>$('previewDialog').close());
  [$('aboutDialog'),$('previewDialog')].forEach(dialog=>dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();}}));
  preview.addEventListener('load',()=>{if(!current&&preview.naturalWidth&&preview.naturalHeight)$('resultDimensions').textContent=`${preview.naturalWidth} × ${preview.naturalHeight}`;});
  preview.addEventListener('error',()=>{if(!current){displayedTrainingUrl=null;preview.hidden=true;$('emptyState').hidden=false;$('emptyState').querySelector('p').textContent='训练预览尚未保存';$('emptyState').querySelector('span').textContent='等待下一次真实模型输出';$('fullscreenButton').disabled=true;}else setMessage('读取到了采样文件，但浏览器无法解码该 PNG。请检查本机服务输出。',true);});
  const pollTimer=setInterval(()=>void loadStatus(),5000);window.addEventListener('beforeunload',()=>{clearInterval(pollTimer);controller?.abort();history.forEach(item=>URL.revokeObjectURL(item.objectUrl));});updatePromptState();syncControls();void loadStatus();
})();
