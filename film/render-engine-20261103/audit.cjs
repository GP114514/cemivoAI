'use strict';
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto');
const {createCanvas,loadImage,fonts,binary}=require('./runtime.cjs');
const ROOT=__dirname,OUT=path.join(ROOT,'renders');
const run=(name,args)=>cp.execFileSync(binary(name),args,{encoding:'utf8',windowsHide:true});
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function main(){
 fonts();const provenance=JSON.parse(fs.readFileSync(path.join(ROOT,'provenance.json'),'utf8'));
 for(const a of provenance.assets)if(sha(fs.readFileSync(path.join(ROOT,a.file)))!==a.sha256)throw Error('Source PNG changed: '+a.file);
 const video=path.join(OUT,'CEMIVO-AI-Rendering-Engine-EN.mp4');
 const p=JSON.parse(run('ffprobe',['-v','error','-count_frames','-show_streams','-show_format','-of','json',video]));
 const v=p.streams.find(s=>s.codec_type==='video'),a=p.streams.find(s=>s.codec_type==='audio');
 if(v.width!==1080||v.height!==1350||v.r_frame_rate!=='30/1'||+v.nb_read_frames!==1200||Math.abs(+p.format.duration-40)>.05||v.codec_name!=='h264'||a.codec_name!=='aac'||+a.sample_rate!==48000)throw Error('Encoded media does not match intended export.');
 const cues=JSON.parse(fs.readFileSync(path.join(ROOT,'audio/cues.json'),'utf8')).cues;
 for(let i=0;i<cues.length;i++)if(cues[i].at+cues[i].duration>39.9||cues[i+1]&&cues[i].at+cues[i].duration>cues[i+1].at-.08)throw Error('Overlapping or truncated narration.');
 const before=createCanvas(256,256),after=createCanvas(256,256);before.getContext('2d').drawImage(await loadImage(path.join(ROOT,'assets/chat-before.png')),0,0);after.getContext('2d').drawImage(await loadImage(path.join(ROOT,'assets/chat-after.png')),0,0);
 const b=before.data(),d=after.data();let outside=0,changed=0;
 for(let y=0;y<256;y++)for(let x=0;x<256;x++){const i=(y*256+x)*4;if(b[i]!==d[i]||b[i+1]!==d[i+1]||b[i+2]!==d[i+2]){changed++;if(!provenance.edit_regions.some(([x0,y0,x1,y1])=>x>=x0&&x<x1&&y>=y0&&y<y1))outside++;}}
 if(outside)throw Error('The illustrative local-edit mask does not match source changes.');
 const times=[1.4,6.6,10.6,16.8,21.5,23.2,25.3,29.2,36.0,39.0];const sheet=createCanvas(1080,3375),ctx=sheet.getContext('2d');ctx.fillStyle='#070c10';ctx.fillRect(0,0,sheet.width,sheet.height);
 for(let i=0;i<times.length;i++){const file=path.join(OUT,`encoded-${i}.png`);run('ffmpeg',['-hide_banner','-loglevel','error','-y','-ss',String(times[i]),'-i',video,'-frames:v','1',file]);ctx.drawImage(await loadImage(file),i%2*540,Math.floor(i/2)*675,540,675);}
 fs.writeFileSync(path.join(OUT,'encoded-storyboard.png'),sheet.toBuffer('image/png'));
 const loudness=cp.spawnSync(binary('ffmpeg'),['-hide_banner','-i',video,'-af','loudnorm=I=-16:TP=-1.5:LRA=7:print_format=json','-f','null','-'],{encoding:'utf8',windowsHide:true});
 const match=loudness.stderr.match(/\{\s*"input_i"[\s\S]*?\}/);const level=match?JSON.parse(match[0]):null;
 // --film is intentionally blocked until a reviewed native recording exists.
 let finalGate=false;try{require('./screen-film.cjs').readRecording(ROOT);}catch(e){finalGate=/REAL SCREEN RECORDING IS STILL MISSING/.test(e.message);}
 const report={kind:'principle_animation_only',video:path.basename(video),sha256:sha(fs.readFileSync(video)),bytes:+p.format.size,duration:+p.format.duration,width:v.width,height:v.height,fps:v.r_frame_rate,frames:+v.nb_read_frames,video_codec:v.codec_name,audio_codec:a.codec_name,narration_clips:cues.length,narration_end:cues.at(-1).at+cues.at(-1).duration,narration_truncated:false,source_PNGs_unchanged:true,illustrated_edit_changed_pixels:changed,illustrated_edit_outside_changed_pixels:outside,simulated_pointer:true,native_screen_recording:false,full_film_missing_recording_guard:finalGate,encoded_keyframes:times,visual_review_pending:true,audio_measurement:level};
 fs.writeFileSync(path.join(ROOT,'audit.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
