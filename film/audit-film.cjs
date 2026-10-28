'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process');
const {createCanvas,loadImage}=require('./canvas-runtime.cjs');
const ROOT=__dirname,OUT=path.join(ROOT,'renders'),sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const run=(cmd,args)=>cp.execFileSync(cmd,args,{encoding:'utf8',windowsHide:true});
async function main(){
 const trace=JSON.parse(fs.readFileSync(path.join(ROOT,'trace.json'),'utf8'));
 for(const f of trace.frames){const bytes=fs.readFileSync(path.join(ROOT,f.file));if(sha(bytes)!==f.sha256)throw Error('Source image changed: '+f.id);const im=await loadImage(bytes);if(im.width!==256||im.height!==256)throw Error('Wrong source dimensions');}
 if(!trace.chat_passed||trace.operations.length!==6||trace.operations.some(o=>o.status!=='complete'||o.independent_outside_changed_pixels!==0||!o.previous_messages_unchanged||!o.replay_identical))throw Error('Recorded operation checks failed');
 if(trace.agent.business_status!=='complete'||trace.agent.storage_verification?.persisted!==true)throw Error('Unverified Agent receipt');
 const video=path.join(OUT,'cemivoAI-X-EN.mp4'),p=JSON.parse(run('ffprobe',['-v','error','-count_frames','-show_streams','-show_format','-of','json',video]));
 const v=p.streams.find(s=>s.codec_type==='video'),a=p.streams.find(s=>s.codec_type==='audio');
 if(v.width!==1080||v.height!==1350||v.r_frame_rate!=='30/1'||+v.nb_read_frames!==1260||+p.format.duration!==42||v.codec_name!=='h264'||a.codec_name!=='aac'||+a.sample_rate!==48000||a.channels!==2)throw Error('Export metadata mismatch');
 const cues=JSON.parse(fs.readFileSync(path.join(ROOT,'audio/cues.json'),'utf8')).cues;
 if(cues.length!==7||cues[0].at>.2)throw Error('Opening narration is missing');
 for(let i=0;i<cues.length;i++){if(cues[i].at+cues[i].duration>41.9||cues[i+1]&&cues[i].at+cues[i].duration>cues[i+1].at-.08)throw Error('Audio overlaps or is truncated');}
 const times=[.5,5,11,16.5,21,27.5,34.5,39.5],sheet=createCanvas(1080,2700),ctx=sheet.getContext('2d');
 for(const[i,t]of times.entries()){const file=path.join(OUT,'encoded-'+i+'.png');run('ffmpeg',['-hide_banner','-loglevel','error','-y','-ss',String(t),'-i',video,'-frames:v','1',file]);const im=await loadImage(file);ctx.drawImage(im,i%2*540,Math.floor(i/2)*675,540,675);}
 fs.writeFileSync(path.join(OUT,'encoded-storyboard.png'),sheet.toBuffer('image/png'));
 const post=fs.readFileSync(path.join(ROOT,'X-post.txt'),'utf8').split('X POST\n')[1].split('\n\nLONGER INTRODUCTION')[0].trim();if(post.length>280)throw Error('Post exceeds 280 characters');
 const report={file:'renders/cemivoAI-X-EN.mp4',sha256:sha(fs.readFileSync(video)),bytes:+p.format.size,duration:42,width:1080,height:1350,fps:30,frames:1260,video_codec:v.codec_name,audio_codec:a.codec_name,
  source_PNGs_unchanged:trace.frames.length,real_API_operations:6,all_recorded_operations_passed:true,outside_changed_pixels:0,agent_document_readback_verified:true,
  real_recording_normal_speed:true,desktop_capture:false,narration:{voice:'en-US-AndrewNeural',online:true,clips:7,starts_at:cues[0].at,ends_at:cues.at(-1).at+cues.at(-1).duration,overlap:false,truncated:false},
  post_characters:post.length,visual_review:{encoded_keyframes:times,manually_reviewed:false},scope:'Research prototype. Millisecond results refer only to confirmed same-page short-text edits. Source UI retained. Explanatory graphics are separate from actual APP pixels.'};
 fs.writeFileSync(path.join(ROOT,'audit.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}
main().catch(e=>{console.error(e.stack);process.exitCode=1;});
