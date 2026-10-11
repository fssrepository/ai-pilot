// Usage: node check-campaign.cjs REPORT.html OR http://127.0.0.1:8090
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const {pathToFileURL}=require('node:url');const path=require('node:path');
(async()=>{
 const target=process.argv[2];if(!target)throw Error('Report path or live URL required');
 const live=/^http:/.test(target), url=live?target:pathToFileURL(path.resolve(target)).href;
 const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1180,height:900}}),errors=[],external=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url())&&(!live||!r.url().startsWith(new URL(url).origin+'/')))external.push(r.url());});
  await page.goto(url);await page.waitForFunction(()=>document.querySelector('#graph').children.length>0);
  const d=await page.evaluate(()=>data);
  if(d.results.length!==12||d.status!=='complete')throw Error('Expected a complete 12-condition report');
  if(await page.locator('#results tr').count()!==12)throw Error('Missing comparison rows');
  await page.selectOption('#run','42|adaptive-replay');
  const events=d.events.filter(e=>e.seed===42&&e.branch==='adaptive-replay');
  const candidate=events.findIndex(e=>e.layers===5);if(candidate<0)throw Error('Actual candidate observation missing');
  await page.locator('#history').fill(String(candidate));await page.locator('#history').dispatchEvent('input');
  if(await page.locator('#follow').isChecked())throw Error('History must disable auto-follow');
  if(await page.locator('#graph .new').textContent()!=='Réteg 5')throw Error('Five-layer candidate graph missing');
  await page.locator('#history').fill('0');await page.locator('#history').dispatchEvent('input');
  if(await page.locator('#graph .new').count())throw Error('Warm start must have four layers');
  if(!await page.locator('#detail').textContent().then(t=>t.includes('Seed 42')))throw Error('Wrong selected observation');
  await page.setViewportSize({width:600,height:850});if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');
  if(errors.length||external.length)throw Error(JSON.stringify({errors,external}));
  if(live){
   for(const route of ['/selected.pt','/../CONFIG']){const r=await fetch(new URL(route,url));if(r.status!==404)throw Error('Viewer exposed an unapproved route');}
   if((await fetch(url,{method:'POST'})).status!==405)throw Error('Viewer accepted mutation');
   // Node fetch normalizes Host; use the HTTP client to actually send this header.
   const hostStatus=await new Promise((resolve,reject)=>require('node:http').get(url,{headers:{Host:'untrusted.example'}},res=>{res.resume();resolve(res.statusCode);}).on('error',reject));
   if(hostStatus!==403)throw Error('Viewer accepted unrelated host');
  }
  if(errors.length||external.length)throw Error(JSON.stringify({errors,external}));
  console.log('PASS:',live?'HTTP viewer':'offline report','12 rows, real four/five-layer history, mobile layout, no script errors/external requests'+(live?', restricted read-only routes':''));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
