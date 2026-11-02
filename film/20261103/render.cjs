'use strict';
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto');
const {once}=require('node:events');
const {createCanvas,loadImage,fonts,binary}=require('./runtime.cjs');
const {draw,W,H,DURATION}=require('./motion.cjs');
const ROOT=__dirname,OUT=path.join(ROOT,'renders'),sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function main(){
 fonts();fs.mkdirSync(OUT,{recursive:true});
 const provenance=JSON.parse(fs.readFileSync(path.join(ROOT,'provenance.json'),'utf8'));
 const images={};for(const a of provenance.assets){const b=fs.readFileSync(path.join(ROOT,a.file));if(sha(b)!==a.sha256)throw Error('Changed source PNG: '+a.file);images[a.id]=await loadImage(b);}
 const cueFile=path.join(ROOT,'audio/cues.json'),cues=fs.existsSync(cueFile)?JSON.parse(fs.readFileSync(cueFile,'utf8')).cues:[];
 const assets={images,cues,editRegions:provenance.edit_regions};
 if(process.argv.includes('--film'))return require('./screen-film.cjs').render({ROOT,assets});
 const canvas=createCanvas(W,H),ctx=canvas.getContext('2d');
 const times=[.8,3,5.7,10.8,15.7,17.4,19.6,24.5];
 const sheet=createCanvas(1080,2700),s=sheet.getContext('2d');
 for(let i=0;i<times.length;i++){draw(ctx,times[i],assets);const buffer=canvas.toBuffer('image/png');fs.writeFileSync(path.join(OUT,`preview-${i}.png`),buffer);s.drawImage(await loadImage(buffer),(i%2)*540,Math.floor(i/2)*675,540,675);}
 fs.writeFileSync(path.join(OUT,'storyboard.png'),sheet.toBuffer('image/png'));
 draw(ctx,1.2,assets);fs.writeFileSync(path.join(OUT,'poster.jpg'),canvas.encodeSync('jpeg',95));
 if(process.argv.includes('--preview')){console.log(JSON.stringify({storyboard:path.join(OUT,'storyboard.png')}));return;}
 if(!process.argv.includes('--principle'))throw Error('Choose --principle (animation only), --preview, or --film (requires genuine screen recording).');
 const output=path.join(OUT,'CEMIVO-Principle-Animation-EN.mp4'),fps=30;
 const audio=path.join(ROOT,'audio/mix.wav');if(!fs.existsSync(audio))throw Error('Generate the voice and mix before rendering.');
 const encoder=cp.spawn(binary('ffmpeg'),['-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pixel_format','rgba','-video_size',`${W}x${H}`,'-framerate',String(fps),'-i','pipe:0','-i',audio,'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-t',String(DURATION),output],{stdio:['pipe','ignore','pipe'],windowsHide:true});
 let log='',closed=false;encoder.stderr.on('data',b=>{log+=b;});encoder.stdin.on('error',()=>{});
 const end=new Promise((resolve,reject)=>{encoder.on('close',code=>{closed=true;code===0?resolve():reject(Error(log));});encoder.on('error',reject);});
 const begin=performance.now();for(let f=0;f<DURATION*fps;f++){if(closed)throw Error('Encoder stopped: '+log);draw(ctx,f/fps,assets);if(!encoder.stdin.write(Buffer.from(canvas.data())))await once(encoder.stdin,'drain');if(f%150===0)console.log(JSON.stringify({frame:f,total:DURATION*fps,elapsed_s:+((performance.now()-begin)/1000).toFixed(1)}));}
 encoder.stdin.end();await end;
 fs.writeFileSync(path.join(OUT,'render.json'),JSON.stringify({file:path.basename(output),sha256:sha(fs.readFileSync(output)),width:W,height:H,fps,duration:DURATION,kind:'principle_animation',desktop_screen_recording:false,simulated_pointer:true,source_PNGs_unchanged:true,final_promotional_film:false,reason:'Live APP screen recording input is still pending.'},null,2));
 console.log(JSON.stringify({complete:true,output,kind:'principle_animation_only'}));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
