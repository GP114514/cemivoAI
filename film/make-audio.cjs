'use strict';
// Original procedural score, mixed under separately generated neural narration.
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const ROOT=__dirname,OUT=path.join(ROOT,'audio'),SR=48000;
const config=JSON.parse(fs.readFileSync(path.join(ROOT,'voice-cues.json'),'utf8').replace(/^\uFEFF/,''));
const D=config.duration,N=Math.round(D*SR),L=new Float32Array(N),R=new Float32Array(N),TAU=Math.PI*2;
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
let seed=429613;function random(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return(seed>>>0)/4294967296;}
function add(at,dur,fn,pan=0){const start=Math.round(at*SR),count=Math.min(Math.round(dur*SR),N-start);const a=Math.sqrt((1-pan)/2),b=Math.sqrt((1+pan)/2);for(let i=0;i<count;i++){const s=fn(i/SR);L[start+i]+=s*a;R[start+i]+=s*b;}}
const chords=[[55,110,130.813,164.814,246.942],[65.406,130.813,164.814,195.998,293.665],[43.654,87.307,130.813,174.614,220],[48.999,97.999,146.832,195.998,246.942]];
for(let at=0,k=0;at<D;at+=6,k++){for(const [j,hz]of chords[k%4].entries())add(at,Math.min(8,D-at),t=>{const env=clamp(t/.4)*Math.exp(-t/5)*clamp((D-at-t)/1.2);return (Math.sin(TAU*hz*t)+.22*Math.sin(TAU*hz*1.002*t))*.018*env;},(j-2)*.25);}
const notes=[220,329.628,440,493.883,261.626,391.995,523.251,659.255];
for(let at=0,k=0;at<D-3;at+=60/106/2,k++){const hz=notes[k%8],amp=at<18?.017:.027;add(at,1.25,t=>Math.sin(TAU*hz*t+.4*Math.sin(TAU*hz*2*t)*Math.exp(-t*12))*Math.exp(-t*7)*clamp(t/.008)*amp,Math.sin(k*1.7)*.65);}
for(let at=0;at<D-5;at+=60/106)add(at,.3,t=>Math.sin(TAU*(46*t+14*.02*(1-Math.exp(-t/.02))))*Math.exp(-t*16)*clamp(t/.004)*(at<18?.015:.028));
for(const at of [0,18,25,31,37])add(at,1.1,t=>(random()*2-1)*.015*Math.sin(Math.PI*t/1.1)**2+Math.sin(TAU*(170+90*t)*t)*.012*Math.exp(-4*t));
const trace=JSON.parse(fs.readFileSync(path.join(ROOT,'trace.json'),'utf8'));
for(const op of trace.operations){const at=(op.t1_ms-trace.frames[0].t_ms)/1000;if(at<18)add(at,.10,t=>(random()*2-1)*.008*Math.exp(-t*80)+Math.sin(TAU*1100*t)*.007*Math.exp(-t*40));}
const probe=file=>+JSON.parse(cp.execFileSync('ffprobe',['-v','error','-show_entries','format=duration','-of','json',file],{encoding:'utf8',windowsHide:true})).format.duration;
const cues=config.cues.map((cue,i)=>({...cue,file:'audio/voice-'+String(i).padStart(2,'0')+'.wav',duration:probe(path.join(OUT,'voice-'+String(i).padStart(2,'0')+'.wav'))}));
for(let i=0;i<cues.length;i++){const c=cues[i];if(c.at+c.duration>D-.10)throw Error('Narration exceeds film: '+i);if(cues[i+1]&&c.at+c.duration>cues[i+1].at-.08)throw Error('Narration overlap: '+i);}
for(let i=0;i<N;i++){const t=i/SR;let duck=1;for(const c of cues){const start=c.at-.08,end=c.at+c.duration+.14;if(t>=start&&t<=end)duck=Math.min(duck,.26+.74*Math.max(clamp((start+.13-t)/.13),clamp((t-end+.13)/.13)));}const gain=clamp(t/.10)*clamp((D-t)/1.5)*duck;L[i]=Math.tanh(L[i]*1.4)*gain;R[i]=Math.tanh(R[i]*1.4)*gain;}
function wav(file){const b=Buffer.alloc(44+N*4);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(2,22);b.writeUInt32LE(SR,24);b.writeUInt32LE(SR*4,28);b.writeUInt16LE(4,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(N*4,40);for(let i=0;i<N;i++){b.writeInt16LE(Math.round(clamp(L[i],-1,1)*32760),44+i*4);b.writeInt16LE(Math.round(clamp(R[i],-1,1)*32760),46+i*4);}fs.writeFileSync(file,b);}
wav(path.join(OUT,'music.wav'));
const args=['-hide_banner','-loglevel','error','-y'];for(const c of cues)args.push('-i',path.join(ROOT,c.file));
const graph=cues.map((c,i)=>`[${i}:a]aresample=48000,aformat=channel_layouts=stereo,highpass=f=70,lowpass=f=15000,adelay=${Math.round(c.at*1000)}|${Math.round(c.at*1000)}[v${i}]`).join(';')+';'+cues.map((_,i)=>`[v${i}]`).join('')+`amix=inputs=${cues.length}:normalize=0,apad,atrim=duration=${D},loudnorm=I=-16:TP=-2:LRA=7[out]`;
cp.execFileSync('ffmpeg',[...args,'-filter_complex',graph,'-map','[out]','-ar','48000','-ac','2',path.join(OUT,'narration.wav')],{windowsHide:true});
cp.execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-i',path.join(OUT,'music.wav'),'-i',path.join(OUT,'narration.wav'),'-filter_complex',`[0:a][1:a]amix=inputs=2:normalize=0,alimiter=limit=0.93,atrim=duration=${D}[out]`,'-map','[out]','-ar','48000','-ac','2',path.join(OUT,'mix.wav')],{windowsHide:true});
fs.writeFileSync(path.join(OUT,'cues.json'),JSON.stringify({...config,cues,music:'Original procedural JavaScript score',narration:'Microsoft en-US-AndrewNeural; generic synthesized narration',offline:false},null,2));
console.log(JSON.stringify({duration:D,cues:cues.map(c=>({at:c.at,duration:c.duration,end:c.at+c.duration})),clipping_and_overlap_guard:'passed'}));
