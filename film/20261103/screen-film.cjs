'use strict';
// The finished cut cannot be exported from API PNGs. A reviewed real screen
// recording, a hash, a visible-viewport crop, and source-time shot ranges are
// required. This module NEVER operates a browser or sends APP requests.
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto');
const {once}=require('node:events');
const {createCanvas,ImageData,binary}=require('./runtime.cjs');
const M=require('./motion.cjs');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function readRecording(ROOT){
 const config=path.join(ROOT,'recordings/recording.json');
 if(!fs.existsSync(config))throw Error('REAL SCREEN RECORDING IS STILL MISSING. See recordings/README.md. The principle animation is not the finished promotional film.');
 const r=JSON.parse(fs.readFileSync(config,'utf8'));
 if(r.capture_kind!=='native_screen_recording'||r.reviewed_for_authenticity!==true||r.reviewed_for_secrets!==true)throw Error('Require an authenticated, privacy-reviewed native screen recording. API output replays are not accepted.');
 if(typeof r.file!=='string'||typeof r.recorded_at!=='string'||!Number.isFinite(Date.parse(r.recorded_at)))throw Error('Require an actual source filename and ISO recording timestamp.');
 const file=path.resolve(ROOT,r.file);if(!fs.existsSync(file)||sha(fs.readFileSync(file))!==r.sha256)throw Error('Screen recording is missing or its SHA256 has changed.');
 const p=JSON.parse(cp.execFileSync(binary('ffprobe'),['-v','error','-show_streams','-show_format','-of','json',file],{encoding:'utf8',windowsHide:true}));
 const v=p.streams.find(s=>s.codec_type==='video');
 if(!v||+p.format.duration<8)throw Error('At least 8 seconds of genuine APP recording is required.');
 if(!r.crop||!['x','y','width','height'].every(k=>Number.isInteger(r.crop[k])))throw Error('Specify an integer source viewport crop.');
 const c=r.crop;if(c.x<0||c.y<0||c.width<=0||c.height<=0||c.x+c.width>v.width||c.y+c.height>v.height)throw Error('Crop must lie inside the source recording.');
 if(!r.shots?.length||r.shots.some(s=>!Number.isFinite(s.start_s)||!Number.isFinite(s.duration_s)||s.start_s<0||s.duration_s<=0||s.start_s+s.duration_s>+p.format.duration))throw Error('Invalid source-time shot ranges.');
 const liveSeconds=r.shots.reduce((a,s)=>a+s.duration_s,0);if(liveSeconds<8||liveSeconds>30)throw Error('Choose 8–30 seconds of normal-speed source footage.');
 for(const p of r.pointer||[])if(!Number.isFinite(p.time_s)||!Number.isFinite(p.x)||!Number.isFinite(p.y)||p.x<0||p.y<0||p.x>1||p.y>1)throw Error('Pointer coordinates must be normalized to the selected crop.');
 if(!r.source_cursor_visible&&!(r.pointer||[]).some(p=>p.click))throw Error('A visible source cursor or a reviewed simulated click track is required.');
 return{...r,file,liveSeconds};
}
async function* decode(recording){
 const {crop:c,file}=recording;
 for(const shot of recording.shots){
  const size=640,filter=`crop=${c.width}:${c.height}:${c.x}:${c.y},scale=${size}:${size}:force_original_aspect_ratio=decrease:flags=neighbor,pad=${size}:${size}:(ow-iw)/2:(oh-ih)/2:color=white,fps=30`;
  const d=cp.spawn(binary('ffmpeg'),['-hide_banner','-loglevel','error','-ss',String(shot.start_s),'-i',file,'-t',String(shot.duration_s),'-an','-vf',filter,'-pix_fmt','rgba','-f','rawvideo','pipe:1'],{stdio:['ignore','pipe','pipe'],windowsHide:true});
  let errors='',buffer=Buffer.alloc(0),index=0;d.stderr.on('data',b=>{errors+=b;});const ended=once(d,'close');ended.catch(()=>{});
  try{
   for await(const chunk of d.stdout){buffer=Buffer.concat([buffer,chunk]);while(buffer.length>=size*size*4){const pixels=Buffer.from(buffer.subarray(0,size*size*4));buffer=buffer.subarray(size*size*4);yield{pixels,size,sourceTime:shot.start_s+index++/30,shot};}}
   const[code]=await ended;if(code!==0)throw Error('Decoder failed: '+errors);
  }finally{if(d.exitCode===null)d.kill();}
 }
}
function pointerAt(track,time){
 const sorted=[...track].sort((a,b)=>a.time_s-b.time_s),prev=[...sorted].reverse().find(p=>p.time_s<=time),next=sorted.find(p=>p.time_s>time);
 if(!prev)return null;const p=next?M.ease((time-prev.time_s)/(next.time_s-prev.time_s)):0;
 const click=[...sorted].reverse().find(p=>p.click&&p.time_s<=time);
 return{x:next?prev.x+(next.x-prev.x)*p:prev.x,y:next?prev.y+(next.y-prev.y)*p:prev.y,age:click?time-click.time_s:-1};
}
async function render({ROOT,assets}){
 const r=readRecording(ROOT),OUT=path.join(ROOT,'renders'),fps=30,total=r.liveSeconds+M.DURATION;
 const canvas=createCanvas(M.W,M.H),ctx=canvas.getContext('2d'),screen=createCanvas(640,640),sctx=screen.getContext('2d');
 const out=path.join(OUT,'CEMIVO-Real-Screen-Film-EN.mp4'),audio=path.join(ROOT,'audio/mix.wav');
 // Audio begins with the principle section. Source audio is omitted to protect
 // unrelated desktop sounds. Live-section narration can be mixed separately.
 const enc=cp.spawn(binary('ffmpeg'),['-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','rgba','-s',`${M.W}x${M.H}`,'-r',String(fps),'-i','pipe:0','-i',audio,'-filter_complex',`[1:a]adelay=${Math.round(r.liveSeconds*1000)}|${Math.round(r.liveSeconds*1000)},apad,atrim=duration=${total}[a]`,'-map','0:v:0','-map','[a]','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-t',String(total),out],{stdio:['pipe','ignore','pipe'],windowsHide:true});
 let logs='',closed=false;enc.stderr.on('data',b=>{logs+=b;});enc.stdin.on('error',()=>{});const finished=new Promise((resolve,reject)=>{enc.on('error',reject);enc.on('close',code=>{closed=true;code===0?resolve():reject(Error(logs));});});finished.catch(()=>{});
 const push=async()=>{if(closed)throw Error('Encoder stopped: '+logs);if(!enc.stdin.write(Buffer.from(canvas.data())))await once(enc.stdin,'drain');};
 try{
 let i=0;for await(const f of decode(r)){
  const t=i++/fps;M.grid(ctx,t);M.text(ctx,'CEMIVO',56,54,23,M.C.ink,true);M.text(ctx,'REAL APP SCREEN RECORDING · 1×',1024,53,15,M.C.blue,false,'right',true);
  M.text(ctx,'Software whose interface',56,175,57,M.C.ink,true);M.text(ctx,'is driven by AI.',56,247,68,M.C.blue,true);
  sctx.putImageData(new ImageData(new Uint8ClampedArray(f.pixels),640,640),0,0);const x=180,y=358,size=720;M.box(ctx,x-14,y-14,size+28,size+28,22,'#fff',null);ctx.imageSmoothingEnabled=false;ctx.drawImage(screen,x,y,size,size);
  const point=pointerAt(r.pointer||[],f.sourceTime);if(point){const cropAspect=r.crop.width/r.crop.height;const dw=cropAspect>1?size:size*cropAspect,dh=cropAspect>1?size/cropAspect:size;M.pointer(ctx,x+(size-dw)/2+point.x*dw,y+(size-dh)/2+point.y*dh,1.3,point.age);}
  M.text(ctx,f.shot.caption||'Recorded from the running APP.',56,1150,28,M.C.ink);M.text(ctx,(r.pointer?.length?'Simulated pointer overlay · ':'')+'Normal-speed segments · waits may be cut',56,1210,22,M.C.muted);M.line(ctx,[[56,1284],[1024,1284]]);M.text(ctx,'ACTUAL RECORDING / '+r.recorded_at.slice(0,10),56,1318,16,M.C.muted,false,'left',true);
  await push();
 }
 const expected=Math.round(r.liveSeconds*fps);if(Math.abs(i-expected)>1)throw Error('Source frame count differs from requested normal-speed duration.');
 for(let j=0;j<M.DURATION*fps;j++){M.draw(ctx,j/fps,assets);await push();}
 enc.stdin.end();await finished;fs.writeFileSync(path.join(OUT,'film-provenance.json'),JSON.stringify({kind:'native_screen_recording_plus_principle_animation',source_sha256:r.sha256,capture_kind:r.capture_kind,source_ranges:r.shots,playback_speed:1,source_frame_content:'crop and scale only; simulated pointer is a labelled overlay',duration:total,source_pointer_visible:r.source_cursor_visible,simulated_pointer:!!r.pointer?.length,sha256:sha(fs.readFileSync(out))},null,2));console.log(JSON.stringify({complete:true,output:out}));
 }catch(error){enc.stdin.destroy();if(!closed)enc.kill();await finished.catch(()=>{});throw error;}
}
module.exports={readRecording,pointerAt,render};
