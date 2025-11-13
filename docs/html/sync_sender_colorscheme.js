const updateColorscheme = () => {
    var colorscheme = document.body.getAttribute("data-md-color-primary");
    if (colorscheme) {
        console.log("Colorscheme changed:", colorscheme);
        localStorage.setItem("colorscheme", colorscheme);
    }
};
document.body.addEventListener("change", updateColorscheme);
document.addEventListener("DOMContentLoaded", updateColorscheme);
