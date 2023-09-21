import { useState } from 'react'
import React, { Component } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState<File | null>(null);
  const [processedImage, setProcessedImage] = useState<string>('');
  const [predictedLabel, setPredictedLabel] = useState<string>('');


  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setImage(e.target.files[0]);
    }
  };

  const submitForm = async (event: React.FormEvent) => {
    // Preventing the page from reloading
    event.preventDefault();
    if (image){
      let form_data = new FormData();
      form_data.append('image', image);
      form_data.append('name', image.name)
      for (var p of form_data) {
        console.log(p);
      }
      let url = 'http://127.0.0.1:8000/api/';
      try {
        const response  = await axios.post(url, form_data,{
        headers:{
          'Content-Type':'multipart/form-data',
        },
      });
      if (response.status === 200){
        setProcessedImage(response.data.image);
        setPredictedLabel(response.data.message);
      }
      else {
        console.log('shiiiit')
      }
          console.log('lets go');
        }
          catch(error){
          console.log(error)
          }
        }
  };

  return (
    <>
    <div className="container">
      <form onSubmit={submitForm}>
        <input
          type="file"
          accept="image/png, image/jpeg"
          onChange={handleImageChange}
          className="input"
        />
        <button type="submit" className="btn">Submit</button>
      </form>
    </div>
    {processedImage && predictedLabel && (
      <div>
        <p>Processed Image:</p>
        <img src={processedImage}/>
        <p>{predictedLabel}</p>
      </div>
   )}
    
    </>
    );
};

export default App;










/*class App extends Component {

  state = {content : ''};

  //const [count, setCount] = useState(0);

  handleSubmit = (e:Event) => {
    e.preventDefault();
    console.log(this.state);
    let form_data = new FormData();
    form_data.append('content', this.state.content);
    let url = 'http://localhost:8000/api/posts/';
    axios.post(url, form_data, {
      headers: {
        'content-type': 'multipart/form-data'
      }
    })
        .then(res => {
          console.log(res.data);
        })
        .catch(err => console.log(err))
  };
  
  return(){

  return (
    
      <div className="App">
      <form className ='imageform' method='POST' onSubmit={this.handleSubmit}>
        <h3>Upload images with React</h3>
        <label>
          Cover
          <input type="file" accept='image/png, image/jpeg' required/>
        </label>
        <br/>
        <button type='submit'>Send Image</button>
        </form>
  </div>
    
  )
}
//}
export default App */
