import React, {useState} from 'react';
function App(){
  const [file,setFile]=useState(null);
  const [preview,setPreview]=useState(null);
  const [result,setResult]=useState(null);
  const onFile=(e)=>{const f=e.target.files[0]; setFile(f); setPreview(URL.createObjectURL(f));}
  const upload=async ()=>{
    if(!file) return; const fd=new FormData(); fd.append('file', file);
    const res=await fetch((process.env.REACT_APP_API_URL || 'http://localhost:8000') + '/predict', {method:'POST', body:fd});
    const data=await res.json(); setResult(data);
  }
  return (
    <div style={{padding:20,fontFamily:'Arial'}}>
      <h2>Potato Disease Detector — Pro</h2>
      <input type='file' accept='image/*' onChange={onFile} />
      {preview && <div style={{marginTop:10}}><img src={preview} alt='preview' style={{maxWidth:300}}/></div>}
      <button onClick={upload} style={{marginTop:10}}>Upload & Predict</button>
      {result && <pre style={{marginTop:20}}>{JSON.stringify(result,null,2)}</pre>}
    </div>
  )
}
export default App;
