// Actual built-in llama.cpp UI, served by the versioned adaptive start-chat.sh.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try {
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8088');
  await page.getByText('Adaptive-Llama-Toy',{exact:true}).first().waitFor();
  const evidence=[];
  for(const [prompt,expected,observed] of [['w=0,x=1,y=2,a=R,b=0>','2,2;','0,2;'],['w=1,x=1,y=2,a=R,b=0>','0,2;','0,2;'],['w=1,x=1,y=2,a=R,b=1>','1,2;','1,2;']]){
   await page.locator('textarea').fill(prompt);
   const pending=page.waitForResponse(r=>r.url().includes('/chat/completions')&&r.request().method()==='POST');
   await page.locator('textarea').press('Enter');
   const response=await pending;const request=response.request().postDataJSON();
   if(response.status()!==200)throw Error(await response.text());
   const body=await response.text();let answer='';
   for(const line of body.split('\n'))if(line.startsWith('data: ')&&!line.includes('[DONE]')){const d=JSON.parse(line.slice(6));answer+=d.choices?.[0]?.delta?.content||'';}
   if(answer!==observed)throw Error(`Unexpected frozen-model response: ${answer}`);
   if(request.max_tokens!==4||request.temperature!==0)throw Error('Built-in UI did not use deterministic four-token settings');
   evidence.push({prompt,expected,predicted:answer,correct:answer===expected,max_tokens:request.max_tokens,temperature:request.temperature});
   await page.waitForFunction(()=>!document.querySelector('textarea').disabled);
  }
  if(errors.length)throw Error(errors.join('\n'));
  console.log(JSON.stringify({passed:true,runtime:'official llama.cpp built-in UI',alias:'Adaptive-Llama-Toy',queries:evidence},null,2));
  await page.screenshot({path:'.runtime/llama-tiny-chat-verified.png',fullPage:true});
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
