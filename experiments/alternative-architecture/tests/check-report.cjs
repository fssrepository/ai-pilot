// Verify recorded graphs, playback, native/browser prediction parity, and viewer boundaries.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const {pathToFileURL}=require('node:url'),path=require('node:path'),fs=require('node:fs');
const {execFileSync}=require('node:child_process');
(async()=>{
 const target=process.argv[2],run=process.argv[3];if(!target||!run)throw Error('Usage: check-report.cjs URL_OR_HTML RUN_DIRECTORY');
 const live=/^http:/.test(target),url=live?target:pathToFileURL(path.resolve(target)).href;
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1180,height:900}}),errors=[],external=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url())&&(!live||!r.url().startsWith(new URL(url).origin+'/')))external.push(r.url());});
  await page.goto(url);await page.waitForFunction(()=>document.querySelector('#results tr'));
  const data=JSON.parse(fs.readFileSync(path.join(run,'status.json')));
  if(data.schema===3){await checkV3(page,data,run);}else{
  if(await page.locator('#results tr').count()!==(data.methods?.length||4)*(data.caps?.length||2))throw Error('Missing method/budget rows');
  const selected=data.results.find(r=>r.seed===(data.schema===2?101:42)&&r.partition==='test'&&r.family==='gate'&&r.gain===2&&r.cap===20000&&r.method===(data.schema===2?'learned-v2':'learned-selector'));
  await page.selectOption('#run',selected.id);
  if(!await page.locator('#graph').textContent().then(x=>x.includes('Kapu b')))throw Error('Actual learned gate missing');
  const expected=JSON.parse(execFileSync('.tools/repair-graph/repair-graph',['predict',path.join(run,selected.id+'.model'),'2','1','0','0','-1'],{encoding:'utf8'})).next_x;
  await page.click('#predict');if(!await page.locator('#prediction').textContent().then(x=>x.endsWith(String(expected))))throw Error('Native/browser prediction mismatch');
  await page.fill('#b','1');await page.click('#predict');if(!await page.locator('#prediction').textContent().then(x=>x.endsWith('2')))throw Error('Learned gate not applied');
  const rejection=data.results.find(r=>r.partition==='test'&&r.events.length>1&&r.events.some(e=>!e.accepted&&!e.exhausted));
  await page.selectOption('#run',rejection.id);
  const rejectedIndex=rejection.events.findIndex(e=>!e.accepted&&!e.exhausted);
  await page.locator('#history').fill(String(rejectedIndex));await page.locator('#history').dispatchEvent('input');
  if(!await page.locator('#detail').textContent().then(x=>x.includes('Elutasítva')))throw Error('Rejected first refit not shown');
  await page.selectOption('#view','candidate');if(!await page.locator('#formula').textContent().then(x=>x.includes('×')))throw Error('Rejected candidate unavailable');
  await page.click('#play');await page.waitForFunction(()=>document.querySelector('#history').value==='1');await page.click('#play');
  await page.selectOption('#view','selected');
  const exhausted=data.results.find(r=>r.events.some(e=>e.exhausted)&&r.partition==='test'&&r.method===(data.schema===2?'learned-v2':'learned-selector'));
  await page.selectOption('#run',exhausted.id);await page.selectOption('#view','candidate');
  if(!await page.locator('#formula').textContent().then(x=>x.includes('részben')))throw Error('Partial candidate presented as usable');
  }
  await page.setViewportSize({width:600,height:850});if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');
  if(live){for(const p of ['/secret.model','/../PROTOCOL.md'])if((await fetch(new URL(p,url))).status!==404)throw Error('Exposed file route');if((await fetch(url,{method:'POST'})).status!==405)throw Error('Accepted mutation');const status=await new Promise((resolve,reject)=>require('node:http').get(url,{headers:{Host:'untrusted.example'}},res=>{res.resume();resolve(res.statusCode);}).on('error',reject));if(status!==403)throw Error('Untrusted host accepted');}
  if(errors.length||external.length)throw Error(JSON.stringify({errors,external}));
  await page.screenshot({path:'.runtime/repair-graph-'+(live?'http':'offline')+'.png',fullPage:true});
  console.log('PASS: '+(live?'HTTP':'offline')+(data.schema===3?' continual routing, learned templates, query history, graph playback, two native prediction formats':' graph history, real accepted/rejected/partial candidates, playback, native prediction parity, gate behavior')+', mobile layout, no external requests/JS errors'+(live?', restricted routes':''));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

async function checkV3(page,data,run){
 if(await page.locator('#results tr').count()!==12)throw Error('Missing continual comparisons');
 const stream=data.streams.find(r=>r.seed===201&&r.method==='bank-macros');
 await page.selectOption('#run',stream.id);
 const expected=JSON.parse(execFileSync('.tools/repair-graph/repair-graph',['predict',path.join(run,stream.id+'.model'),'2','1','0','0','-1'],{encoding:'utf8'}));
 await page.click('#predict');if(!await page.locator('#prediction').textContent().then(s=>s.endsWith(String(expected.next_x))))throw Error('Continual native/browser mismatch');
 const returned=stream.windows.findIndex(w=>w.phase==='return'&&w.reused);
 await page.locator('#history').fill(String(returned));await page.locator('#history').dispatchEvent('input');
 if(!await page.locator('#detail').textContent().then(s=>s.includes('Korábbi gráf automatikusan')))throw Error('Automatic return missing');
 if(!await page.locator('#v3-evidence').textContent().then(s=>s.includes('forrásgráfok:')&&s.includes('c2')))throw Error('Learned supports/holes missing');
 await page.selectOption('#view','before');
 const before=await page.evaluate(()=>currentGraph);if(JSON.stringify(before)!==JSON.stringify(stream.windows[returned].before))throw Error('Wrong prior graph');
 await page.selectOption('#view','selected');
 await page.click('#play');await page.waitForFunction(()=>document.querySelector('#history').value==='1');await page.click('#play');
 await page.selectOption('#experiment','inquiry');
 const inquiry=data.inquiries.find(r=>r.seed===201&&r.family===3&&r.gain===-2&&r.method==='active'&&r.cap===8);
 await page.selectOption('#run',inquiry.id);
 const native=JSON.parse(execFileSync('.tools/repair-graph/repair-graph',['predict-inquiry',path.join(run,inquiry.id+'.inquiry'),String(inquiry.clean_source),'2','1','0','0','-1'],{encoding:'utf8'}));
 await page.click('#predict');if(!await page.locator('#prediction').textContent().then(s=>s.endsWith(String(native.next_x))))throw Error('Inquiry native/browser mismatch');
 const contradiction=data.inquiries.find(r=>r.method==='active'&&r.events.some(e=>e.after_hypotheses===0));
 await page.selectOption('#run',contradiction.id);await page.locator('#history').fill(String(contradiction.events.findIndex(e=>e.after_hypotheses===0)));await page.locator('#history').dispatchEvent('input');
 if(!await page.locator('#v3-evidence').textContent().then(s=>s.includes('nem magyarázza')))throw Error('Unsupported-source explanation missing');
 if(await page.locator('#results tr').count()!==12)throw Error('Missing inquiry comparisons');
 await page.fill('#a','9');await page.click('#predict');if(!await page.locator('#prediction').textContent().then(s=>s.includes('határok')))throw Error('Input bounds missing');
}
