const captcha_input = document.getElementById("captcha");
const canvas = document.createElement("canvas");
var refresh_btn = document.getElementById("loginCapchaRefresh");
const submit_btn = document.getElementById("submit");

reading = () => {
    image = document.getElementById("appCaptchaLoginImg");
    image2 = document.createElement("img");
    image2.src = image.src;
    canvas.width = 150;
    canvas.height = 50;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image2, 0, 0);

    const img_data = canvas.toDataURL();
    api_fetch(img_data);
}

const api_fetch = async (img_data) => {
    api_url = "https://aims-captcha-reader-api.onrender.com";
    const response = await fetch(api_url, {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ "image": img_data })
    })
    const result = await response.json();

    captcha_input.value = result.captcha_value

    submit_btn.click()
}

setTimeout(reading, 1000);

refresh_btn.addEventListener("click", () => {
    setTimeout(reading, 1000);
})