'use strict';
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),{once}=require('node:events');
const {createCanvas,loadImage,GlobalFonts}=require('./canvas-runtime.cjs');
const {drawFilm,DURATION,WIDTH,HEIGHT}=require('./composer.cjs');
const ROOT=__dirname,OUT=path.join(ROOT,'renders');fs.mkdirSync(OUT,{recursive:true});
GlobalFonts.registerFromPath('C:/Windows/Fonts/segoeui.ttf','Cemivo Sans');GlobalFonts.registerFromPath('C:/Windows/Fonts/segoeuib.ttf','Cemivo Bold');GlobalFonts.registerFromPath('C:/Windows/Fonts/consola.ttf','Consolas');
async function main(){
 const trace=JSON.parse(fs.readFileSync(path.join(ROOT,'trace.json'),'utf8'));const cueFile=path.join(ROOT,'audio/cues.json');if(fs.existsSync(cueFile)){const audio=JSON.parse(fs.readFileSync(cueFile,'utf8'));trace.voice_cues=audio.cues;}
 else trace.voice_cues=[];
 const images={};for(const frame of trace.frames)images[frame.id]=await loadImage(path.join(ROOT,frame.file));const assets={trace,images};const canvas=createCanvas(WIDTH,HEIGHT),ctx=canvas.getContext('2d');
 const times=[.5,5,11,16.5,21,27.5,34.5,39.5],sheet=createCanvas(1080,2700),sctx=sheet.getContext('2d');
 for(const[i,t]of times.entries()){drawFilm(ctx,t,assets);const png=canvas.toBuffer('image/png');fs.writeFileSync(path.join(OUT,'preview-'+String(i).padStart(2,'0')+'.png'),png);const snapshot=await loadImage(png);sctx.drawImage(snapshot,i%2*540,Math.floor(i/2)*675,540,675);}
 fs.writeFileSync(path.join(OUT,'storyboard.png'),sheet.toBuffer('image/png'));drawFilm(ctx,.5,assets);fs.writeFileSync(path.join(OUT,'poster.jpg'),canvas.encodeSync('jpeg',96));
 if(process.argv.includes('--preview')){console.log(JSON.stringify({preview:true,frames:times.length,storyboard:path.join(OUT,'storyboard.png')}));return;}
 const output=path.join(OUT,'cemivoAI-X-EN.mp4'),fps=30;
 const args=['-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pixel_format','rgba','-video_size',WIDTH+'x'+HEIGHT,'-framerate',String(fps),'-i','pipe:0','-i',path.join(ROOT,'audio/mix.wav'),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-t',String(DURATION),output];
 const encoder=cp.spawn('ffmpeg',args,{stdio:['pipe','ignore','pipe'],windowsHide:true});let log='';encoder.stderr.on('data',b=>{log+=b;});let closed=false;const finished=new Promise((resolve,reject)=>{encoder.on('close',code=>{closed=true;code===0?resolve():reject(Error('FFmpeg exited '+code+': '+log));});encoder.on('error',reject);});encoder.stdin.on('error',()=>{});const start=performance.now();
 for(let i=0;i<DURATION*fps;i++){if(closed)throw Error('Encoder closed early: '+log);drawFilm(ctx,i/fps,assets);if(!encoder.stdin.write(Buffer.from(canvas.data())))await once(encoder.stdin,'drain');if(i%150===0)console.log(JSON.stringify({frame:i,total:DURATION*fps,elapsed_s:+((performance.now()-start)/1000).toFixed(1)}));}
 encoder.stdin.end();await finished;fs.writeFileSync(path.join(OUT,'render-log.txt'),log);fs.writeFileSync(path.join(OUT,'render-metadata.json'),JSON.stringify({duration:DURATION,fps,width:WIDTH,height:HEIGHT,voice_cues:trace.voice_cues},null,2));console.log(JSON.stringify({complete:true,output,duration:DURATION,fps,width:WIDTH,height:HEIGHT}));
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
