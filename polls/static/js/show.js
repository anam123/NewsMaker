function showhide()
 {

       var div = document.getElementById("newpost");
    
       var div1 = document.getElementById("summary");
  
if (div.style.display !== "none") {
    div.style.display = "none";
    div1.style.display = "block";

}
else {
    div.style.display = "block";
    div1.style.display="none";
    
}

 }