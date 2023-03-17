alert("test")

const inputFile = document.getElementById("fileUpload");

async function sendFiles(){
  const formData = new FormData();
  try
  {
    for (const file of inputFile.files) {
      formData.append("images", file);
    }
    
    console.log(formData)
    
    const resp = await fetch(
      "http://127.0.0.1:5000/upload",
      {
        method:"POST",
        body:formData,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );

    // const jsn = await resp.json();
    // console.log(jsn);
  }
  catch(err)
  {
    console.log(err);
  }
};

sendFiles();