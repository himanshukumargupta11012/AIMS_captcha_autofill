image = document.getElementById("appCaptchaLoginImg")
refresh = document.getElementById("loginCapchaRefresh")

j = 0;
inter = setInterval(() => {
 i = image.src.length - 1
 while (image.src[i] != '/') i--;
 
 link = document.createElement("a")
 link.href = image.src
 link.download = image.src.slice(i + 1) + ".png"

 document.body.appendChild(link)
 link.click()

 document.body.removeChild(link)

 refresh.click()
 j++;

 if(j == 5000) clearInterval(inter)
}, 200);




