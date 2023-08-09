if (window.location.href == "https://aims.iith.ac.in/aims/") {

captcha_input = document.getElementById("captcha")
var refresh_btn = document.getElementById("loginCapchaRefresh");
const login_btn = document.getElementById("login");
const input = document.getElementById("uid");
const pwd_input = document.getElementById("pswrd");

reading = ()=>{
    image = document.getElementById("appCaptchaLoginImg")
    i = image.src.length - 1
    while (image.src[i] != '/') i--;
    captcha_val = image.src.slice(i + 1)
    captcha_input.value = captcha_val;
    input.value = "ai21btech11012";
    pwd_input.value = "TWebM6p7";

    setTimeout(() => { login_btn.click(); }, 100)
}


reading();

refresh_btn.addEventListener("click", () => {
    setTimeout(() => { reading(); }, 100);
})

}