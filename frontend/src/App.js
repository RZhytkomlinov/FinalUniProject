import React from "react";

class Main extends React.Component{
  
  handleclick(){

  }

  render() {
    return(
      <button src={this.state.img} onClick={this.handleclick}>Upload</button>
    )
  }
}

export default Main;
