if (window.location.href == "https://aims.iith.ac.in/aims/") {

captcha_input = document.getElementById("captcha")
var refresh_btn = document.getElementById("loginCapchaRefresh");

reading = ()=>{
    image = document.getElementById("appCaptchaLoginImg")
    i = image.src.length - 1
    while (image.src[i] != '/') i--;
    captcha_val = image.src.slice(i + 1)
    captcha_input.value = captcha_val;
    // input.value = "";
    // pwd_input.value = "";

    // setTimeout(() => { login_btn.click(); }, 100)
}


reading();

refresh_btn.addEventListener("click", () => {
    setTimeout(() => { reading(); }, 100);
})

}