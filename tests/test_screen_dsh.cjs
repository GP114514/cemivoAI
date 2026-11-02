const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(path.join(__dirname,'../web/screen.js'),'utf8');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const frame={session_id:'b'.repeat(32),version:1,width:256,height:256,sha256:'c'.repeat(64),image_url:'/api/live/images/'+ 'a'.repeat(32)+'.png'};

function harness({fail=false,verified=true,protocol=1,privatePage=false}={}){
 const nodes={},calls=[],stored=new Map(),intervals=[];
 const scene=privatePage?{...frame,credential_request:{kind:'wifi_password',token:'d'.repeat(48),
  input_bounds:[12,112,240,134],submit_bounds:[12,150,112,180]}}:frame;
 let configured=false,backend='deepseek',finished=false;
 function el(id){return nodes[id]||={id,hidden:!['setup','setupForm','setupEnter','setupBackend'].includes(id),value:id==='setupBackend'?'dsh':'',textContent:'',disabled:false,style:{},events:{},addEventListener(k,f){this.events[k]=f;},focus(){},getBoundingClientRect(){return {width:256,height:256,left:0,top:0};}};}
 const ok=value=>({ok:true,json:async()=>value});
 const context={document:{getElementById:el,body:{dataset:{}}},window:{addEventListener(){}},
  sessionStorage:{getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v),removeItem:k=>stored.delete(k)},
  setTimeout:(...args)=>{const timer=setTimeout(...args);timer.unref();return timer;},clearTimeout,
  setInterval:fn=>intervals.push(fn),AbortController,TextEncoder,URL,
  location:{origin:'http://127.0.0.1:7861'},crypto:{randomUUID:()=> 'pending-request-001'},
  Image:class {naturalWidth=256;naturalHeight=256;async decode(){}},
  fetch:async(url,opt)=>{
   const body=opt.body?JSON.parse(opt.body):null;calls.push({url,body});
   if(url==='/api/status')return ok({dsh_graphics_protocol:protocol});
   if(url==='/api/live/status')return ok({planner:{configured,backend},runtime:{busy:false}});
   if(url==='/api/live/config'){
    if(fail)return {ok:false,status:503,json:async()=>({code:'dsh_unavailable',error:'DSH unavailable'})};
    configured=true;backend=body.backend;return ok({configured,backend,verified});
   }
   if(url==='/api/live/current')return ok(scene);
   if(url==='/api/computer/wifi-credentials')return ok({...frame,version:2,agent:{status:'complete',answer:'Actual local result'}});
   if(url==='/api/agent/event')return ok({...frame,agent:{status:'awaiting_result',dsh_pending:true,warning:'Waiting for the actual DSH result.'}});
   if(url.startsWith('/api/agent/request?'))return ok({status:finished?'render_failed':'awaiting_result',warning:finished?'DSH finished; actual receipts updated.':undefined});
   throw Error('Unexpected fixture endpoint '+url);
  }};
 vm.runInNewContext(source,context);
 return {el,calls,stored,intervals,finish:()=>{finished=true;}};
}

async function setupCases(){
 for(const options of [{},{fail:true},{verified:false},{protocol:0}]){
  const h=harness(options);await tick();
  assert(!h.calls.some(c=>c.url==='/api/live/current'));
  await h.el('setupForm').events.submit({preventDefault(){}});await tick();
  const valid=!options.fail&&options.verified!==false&&options.protocol!==0;
  assert.equal(h.el('setup').hidden,valid);
  assert.equal(h.calls.some(c=>c.url==='/api/live/current'),valid);
  const config=h.calls.find(c=>c.url==='/api/live/config');
  if(options.protocol===0)assert(!config);else assert.deepEqual(config.body,{backend:'dsh'});
  assert(!h.el('setupEnter').disabled);
 }
 const h=harness();await tick();h.el('setupBackend').value='deepseek';h.el('setupBackend').events.change();
 assert.equal(h.el('setupKey').hidden,false);
 await h.el('setupForm').events.submit({preventDefault(){}});
 assert(!h.calls.some(c=>c.url==='/api/live/config'));
 h.el('setupKey').value='fixture-not-a-real-key';
 await h.el('setupForm').events.submit({preventDefault(){}});await tick();
 assert(h.el('setup').hidden);assert.equal(h.el('setupKey').value,'');
 assert(![...h.stored.values()].some(value=>value.includes('fixture-not-a-real-key')));
}

async function pendingInteraction(){
 const h=harness();await tick();await h.el('setupForm').events.submit({preventDefault(){}});await tick();
 assert.equal(h.el('image').hidden,false);
 h.el('image').events.click({clientX:128,clientY:128});await tick();await tick();
 assert.equal(h.stored.get('cemivoAI.screen.pending'),'pending-request-001');
 h.el('image').events.click({clientX:140,clientY:140});await tick();
 assert.equal(h.calls.filter(c=>c.url==='/api/agent/event').length,1);
 await h.intervals[0]();assert(h.stored.has('cemivoAI.screen.pending'));
 h.finish();await h.intervals[0]();
 assert(!h.stored.has('cemivoAI.screen.pending'));
 assert.equal(h.calls.filter(c=>c.url==='/api/agent/event').length,1);
 assert(h.el('notice').textContent.includes('actual receipts updated'));
}

async function privateInputStaysInTheGeneratedScreen(){
 const h=harness({privatePage:true});await tick();await h.el('setupForm').events.submit({preventDefault(){}});await tick();
 assert.equal(h.el('wifiPassword').hidden,false);
 assert.equal(h.el('wifiPassword').style.left,'12px');
 h.el('wifiPassword').value='LOCAL_ONLY_TEST_8391';
 h.el('wifiPassword').events.keydown({key:'Enter',preventDefault(){},isComposing:false,repeat:false});
 await tick();await tick();
 const submitted=h.calls.filter(c=>c.url==='/api/computer/wifi-credentials');assert.equal(submitted.length,1);
 assert.equal(submitted[0].body.password,'LOCAL_ONLY_TEST_8391');
 assert(!h.calls.some(c=>c.url==='/api/agent/event'));
 assert.equal(h.el('wifiPassword').value,'');
 assert(![...h.stored.values()].some(v=>String(v).includes('LOCAL_ONLY_TEST_8391')));
 assert.equal(h.el('text').value,'');
}

(async()=>{await setupCases();await pendingInteraction();await privateInputStaysInTheGeneratedScreen();console.log('PASS: DSH setup, pending deduplication and in-image private password submission without model/storage leakage.');})().catch(error=>{console.error(error);process.exitCode=1;});
