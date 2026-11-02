'use strict';
// JS vector diagrams explain the architecture. PNG panels are unchanged real
// model outputs. Tensor patterns are illustrative, not captured activations.
const W=1080,H=1350,DURATION=40;
const C={bg:'#070c10',ink:'#f3f7f4',muted:'#7e9b92',line:'#243a35',blue:'#a1f4cc',paper:'#edf3ee',black:'#12211b',gray:'#687e71'};
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const ease=v=>1-(1-clamp(v))**3;
const smooth=v=>{v=clamp(v);return v*v*(3-2*v);};
const mix=(a,b,p)=>a+(b-a)*p;
const noise=n=>{const x=Math.sin(n*137.1+72.7)*43758.5453;return x-Math.floor(x);};
function text(c,s,x,y,size=26,color=C.ink,bold=false,align='left',mono=false){c.font=`${bold?700:400} ${size}px ${mono?'"Film Mono", monospace':bold?'"Film Bold", sans-serif':'"Film Sans", sans-serif'}`;c.textAlign=align;c.textBaseline='alphabetic';c.fillStyle=color;c.fillText(String(s),x,y);}
function box(c,x,y,w,h,r=18,fill='#101d19',stroke=C.line){c.beginPath();c.roundRect(x,y,w,h,r);if(fill){c.fillStyle=fill;c.fill();}if(stroke){c.strokeStyle=stroke;c.lineWidth=1;c.stroke();}}
function line(c,points,color=C.line,width=1){c.beginPath();c.moveTo(...points[0]);for(const p of points.slice(1))c.lineTo(...p);c.strokeStyle=color;c.lineWidth=width;c.stroke();}
function dot(c,x,y,r=4,color=C.blue){c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fillStyle=color;c.fill();}
function glow(c,x,y,r,alpha=.12){const g=c.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,`rgba(90,220,165,${alpha})`);g.addColorStop(1,'rgba(90,220,165,0)');c.fillStyle=g;c.fillRect(x-r,y-r,r*2,r*2);}
function grid(c,t){c.fillStyle=C.bg;c.fillRect(0,0,W,H);glow(c,820,500,780,.08);for(let x=52;x<W;x+=42)for(let y=100;y<H-70;y+=42){c.fillStyle=`rgba(160,240,200,${.035+.015*Math.sin(x*.013+y*.004+t*.3)})`;c.fillRect(x,y,1,1);}}
function source(c,img,x,y,size){c.save();c.imageSmoothingEnabled=false;c.drawImage(img,x,y,size,size);c.restore();}
function tag(c,s,x,y){c.font='400 17px "Film Mono"';const w=c.measureText(s).width+28;box(c,x,y,w,34,17,'#152b22',null);text(c,s,x+14,y+23,17,C.blue,false,'left',true);}
function wrapped(c,s,x,y,w,size=27,color=C.ink,leading=size*1.3,align='left'){c.font=`400 ${size}px "Film Sans", sans-serif`;const rows=[];let row='';for(const word of String(s).split(/\s+/)){const next=row?row+' '+word:word;if(c.measureText(next).width>w&&row){rows.push(row);row=word;}else row=next;}if(row)rows.push(row);rows.forEach((r,i)=>text(c,r,x,y+i*leading,size,color,false,align));return rows;}
function title(c,a,b){text(c,a,56,168,65,C.ink,true);text(c,b,56,244,65,C.blue,true);}
function pointer(c,x,y,scale=1,age=-1){c.save();c.translate(x,y);c.scale(scale,scale);if(age>=0&&age<.8){c.globalAlpha=1-age/.8;c.strokeStyle=C.blue;c.lineWidth=2;c.beginPath();c.arc(0,0,8+age*65,0,Math.PI*2);c.stroke();c.globalAlpha=1;}c.beginPath();c.moveTo(0,0);c.lineTo(0,30);c.lineTo(8,23);c.lineTo(15,37);c.lineTo(22,34);c.lineTo(15,20);c.lineTo(27,20);c.closePath();c.fillStyle='#fff';c.strokeStyle='#17261f';c.lineWidth=2;c.shadowColor='#0008';c.shadowBlur=8;c.fill();c.shadowBlur=0;c.stroke();c.restore();}
function pulse(c,points,p){const lengths=points.slice(1).map((v,i)=>Math.hypot(v[0]-points[i][0],v[1]-points[i][1]));let d=clamp(p)*lengths.reduce((a,b)=>a+b,0);for(let i=0;i<lengths.length;i++){if(d<=lengths[i]||i===lengths.length-1){const q=d/lengths[i],x=mix(points[i][0],points[i+1][0],q),y=mix(points[i][1],points[i+1][1],q);glow(c,x,y,30,.16);dot(c,x,y);return;}d-=lengths[i];}}
function chrome(c,t,name){text(c,'CEMIVO',56,53,23,C.ink,true);text(c,'AI RENDER ENGINE / PRINCIPLE FILM',1024,52,15,C.muted,false,'right',true);line(c,[[56,79],[1024,79]]);line(c,[[56,1284],[1024,1284]]);line(c,[[56,1284],[56+968*t/DURATION,1284]],C.blue,3);text(c,name,56,1319,15,C.muted,false,'left',true);text(c,`${String(Math.floor(t)).padStart(2,'0')} / ${DURATION}`,1024,1319,15,C.muted,false,'right',true);}
function captions(c,t,a){const cue=a.cues?.find(q=>t>=q.at&&t<q.at+q.duration);if(!cue)return;box(c,40,1152,1000,110,18,'#070c10ef',null);wrapped(c,cue.text,540,1194,910,29,'#e5f1e8',38,'center');}
function imagePanel(c,img,x,y,s){box(c,x-12,y-12,s+24,s+24,18,'#fff',null);source(c,img,x,y,s);}
function reveal(c,img,x,y,s,p){const rows=16,cell=s/rows;for(let yy=0;yy<rows;yy++)for(let xx=0;xx<rows;xx++){const n=noise(xx+yy*rows);if(n<p){c.save();c.beginPath();c.rect(x+xx*cell,y+yy*cell,cell+.3,cell+.3);c.clip();source(c,img,x,y,s);c.restore();}else{c.fillStyle=`rgba(161,244,204,${.03+.08*n})`;c.fillRect(x+xx*cell,y+yy*cell,cell-1,cell-1);}}}

function opening(c,t,a){
 title(c,'AI as a software','rendering engine.');
 text(c,'The interface itself is generated output.',58,301,29,C.muted);
 const x=257,y=367,s=566,p=ease((t-.25)/2.2);
 box(c,x-13,y-13,s+26,s+26,22,'#15231e','#385f4c');
 const img=t<3.8?a.images.settings:t<6.2?a.images.chatBefore:a.images.chatAfter;
 reveal(c,img,x,y,s,p);
 tag(c,'MODEL OUTPUT / 256 × 256',56,1011);
 text(c,'Intent in. A working interface out.',56,1093,32,C.ink,true);
 if(t>=3.8){const m=smooth((t-4.1)/1.2);pointer(c,mix(940,x+s*201/256,m),mix(1040,y+s*225/256,m),1.1,t-6.1);}
}

function renderLoop(c,t,a){
 title(c,'Software, rendered','from intent.');
 text(c,'A new rendering request with every interaction.',56,300,27,C.muted);
 const x=84,y=373,s=412;imagePanel(c,a.images.settings,x,y,s);
 const p=ease((t-.1)/1.2);pointer(c,mix(510,x+s*.73,p),mix(842,y+s*.35,p),1.15,t-1.4);
 const rows=[['SCREEN','The current image'],['INPUT','Words + click position'],['STATE','What has happened']];
 rows.forEach(([label,sub],i)=>{const yy=395+i*133;box(c,609,yy,415,106,16);text(c,label,635,yy+35,18,C.blue,false,'left',true);text(c,sub,635,yy+78,26,C.ink);line(c,[[520,yy+53],[609,yy+53]],C.line,2);pulse(c,[[520,yy+53],[609,yy+53]],(t*.55+i*.28)%1);});
 const flow=[[286,817],[286,889],[540,889],[540,954]];line(c,flow,C.line,2);pulse(c,flow,(t*.3)%1);
 box(c,56,954,968,132,22,'#162f24','#406951');text(c,'NEXT RENDER REQUEST',88,1001,20,C.blue,false,'left',true);text(c,'Full-frame context → intent → next pixels',88,1054,34,C.ink,true);
}

function neural(c,t,a){
 title(c,'AI plans the layout.','Neural heads draw.');
 text(c,'A learned rendering layer inside the software.',56,302,28,C.muted);
 const sx=56,sy=398,ss=426,dx=598,dy=398,ds=426;
 tag(c,'01 / LAYOUT',sx,342);tag(c,'02 / MODEL OUTPUT',dx,342);
 box(c,sx-8,sy-8,ss+16,ss+16,16,'#0d1b16','#44634f');
 for(let i=0;i<a.layout.elements.length;i++){
  const e=a.layout.elements[i],b=e.bounds,p=ease((t-i*.035)/.75);c.save();c.globalAlpha=.2+.8*p;
  const x=sx+b[0]/256*ss,y=sy+b[1]/256*ss,w=(b[2]-b[0])/256*ss,h=(b[3]-b[1])/256*ss;
  box(c,x,y,w,h,2,null,e.kind==='text'?'#395344':C.blue);c.restore();
 }
 text(c,'DSH chooses elements and bounds',sx,872,21,C.muted);
 box(c,dx-8,dy-8,ds+16,ds+16,16,'#14291e','#44634f');reveal(c,a.images.settings,dx,dy,ds,ease((t-.8)/2.1));
 text(c,'Actual trained-decoder output',dx,872,21,C.muted);
 const arrow=[[504,611],[576,611]];line(c,arrow,C.line,2);pulse(c,arrow,(t*.65)%1);
 const layers=[['SURFACE HEAD',56],['TEXT HEAD',389],['PIXEL FRAME',722]];
 layers.forEach(([s,x],i)=>{box(c,x,947,302,107,16,i===2?'#1b3a2b':'#101e18',i===2?'#5a9874':C.line);text(c,s,x+22,983,17,C.blue,false,'left',true);for(let n=0;n<22;n++){const alpha=.12+.7*noise(n+i*44+Math.floor(t*6));c.fillStyle=`rgba(161,244,204,${alpha})`;c.fillRect(x+23+n*11,1004,7,27*noise(n+5*i));}});
 text(c,'Layout diagram and decoder symbols are explanatory.',56,1111,23,C.muted);
}

function scope(c,t,a){
 title(c,t<3.8?'Re-render a region.':'Render a new view.',t<3.8?'Preserve the rest.':'Follow the next goal.');
 const isLocal=t<3.8;
 text(c,isLocal?'Same-page change → bounded pixel update':'New page → AI-selected layout and new pixels',56,301,27,C.muted);
 const x=72,y=391,s=580;
 if(isLocal){imagePanel(c,t<1.9?a.images.chatBefore:a.images.chatAfter,x,y,s);
  for(const b of a.editRegions){c.strokeStyle=C.blue;c.lineWidth=2.5;c.strokeRect(x+b[0]*s/256,y+b[1]*s/256,(b[2]-b[0])*s/256,(b[3]-b[1])*s/256);}
  pointer(c,x+s*201/256,y+s*225/256,1.2,t-1.5);
 }else{imagePanel(c,a.images.settings,x,y,s);}
 tag(c,isLocal?'LOCAL RENDER':'GLOBAL RENDER',731,405);
 const info=isLocal?[['0','outside pixels changed'],['2','selected edit regions']]:[['AI','chooses the controls'],['256²','native output pixels']];
 info.forEach(([v,label],i)=>{const yy=550+i*240;text(c,v,731,yy,isLocal?100:85,C.blue,true);wrapped(c,label,731,yy+48,280,25,C.muted,34);});
 text(c,isLocal?'Measured from the retained before/after model PNGs.':'New-view example is an actual generated Settings frame.',56,1087,24,C.muted);
 text(c,'Source images preserved · replay timing is illustrative',56,1125,21,C.muted);
}

function agent(c,t,a){
 title(c,'The agent acts.','The engine renders.');
 text(c,'Task results feed the next visual state.',56,300,29,C.muted);
 const rows=[['AGENT','Executes an actual tool','Tools / files / native actions'],['RESULT','Returns verified state','Success, failure or pending'],['RENDER ENGINE','Turns state into the next UI','AI layout → neural pixel output']];
 rows.forEach(([label,headline,sub],i)=>{const y=373+i*223;box(c,56,y,968,179,23,i===2?'#1a3427':'#101e18',i===2?'#558b6c':C.line);text(c,label,85,y+40,18,C.blue,false,'left',true);text(c,headline,85,y+98,39,C.ink,true);text(c,sub,85,y+147,26,C.muted);if(i<2){const p=[[540,y+179],[540,y+223]];line(c,p,C.line,2);pulse(c,p,(t*.6+i*.3)%1);}});
 text(c,'Architecture diagram · no task execution claimed in this scene',56,1105,22,C.muted);
}

function closing(c,t,a){
 const p=ease(t/.8);c.save();c.globalAlpha=p;glow(c,540,625,520,.10);
 text(c,'AI IS THE',540,267,70,C.ink,true,'center');text(c,'RENDERING LAYER.',540,351,70,C.blue,true,'center');
 const labels=['INTENT','AI LAYOUT','PIXELS','INTERACTION'],pos=[[280,514],[800,514],[800,795],[280,795]];
 const edges=[[[280,514],[800,514]],[[800,514],[800,795]],[[800,795],[280,795]],[[280,795],[280,514]]];
 edges.forEach((e,i)=>{line(c,e,C.line,2);pulse(c,e,(t*.28+i*.25)%1);});labels.forEach((s,i)=>{const[x,y]=pos[i];box(c,x-125,y-34,250,68,34,'#10231a',C.line);text(c,s,x,y+8,20,C.blue,false,'center',true);});
 text(c,'CEMIVO',540,679,86,C.ink,true,'center');text(c,'An interface that becomes what comes next.',540,934,32,C.ink,false,'center');text(c,'github.com/GP114514/cemivoAI',540,1013,27,C.blue,false,'center',true);text(c,'Open source · Research prototype · Principle animation',540,1085,23,C.muted,false,'center');c.restore();
}

const SCENES=[{at:0,name:'01 / AI AS THE ENGINE',draw:opening},{at:8.4,name:'02 / THE RENDER REQUEST',draw:renderLoop},{at:14.3,name:'03 / LEARNED PIXEL DECODING',draw:neural},{at:20.2,name:'04 / LOCAL + GLOBAL RENDER',draw:scope},{at:27,name:'05 / ACTION → STATE → PIXELS',draw:agent},{at:33.4,name:'06 / CEMIVO',draw:closing}];
function draw(c,time,a){const t=clamp(time,0,DURATION-.0001),scene=[...SCENES].reverse().find(s=>t>=s.at);c.save();c.resetTransform();c.globalAlpha=1;c.clearRect(0,0,W,H);grid(c,t);scene.draw(c,t-scene.at,a);chrome(c,t,scene.name);captions(c,t,a);const age=t-scene.at;if(age<.18&&scene.at>0){c.fillStyle=`rgba(7,12,16,${1-ease(age/.18)})`;c.fillRect(0,82,W,1060);}c.restore();}
module.exports={draw,W,H,DURATION,SCENES,pointer,text,box,source,line,grid,clamp,ease,C};
