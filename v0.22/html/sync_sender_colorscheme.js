const updateImageColorsheme = () => {
    const isSmallScreen = window.innerWidth < 1280;
    const images = document.querySelectorAll('.logo-img');
    const color = localStorage.getItem('colorscheme');
    images.forEach(img => {
        const isDarkImage = img.src.includes('#only-dark');
        // case of default theme is white is not handled well
        const shouldHide = isSmallScreen && (
            (!isDarkImage && color === "black") ||
            (isDarkImage && color !== "black")
        );
        img.style.display = shouldHide ? 'none' : '';
    });
    const menuSubtitle = document.querySelector(".md-nav__title");
    if (isSmallScreen) {
        menuSubtitle.removeChild(menuSubtitle.childNodes[2]);
        menuSubtitle.style.height = "auto";
    }
}
const updateColorscheme = () => {
    var colorscheme = document.body.getAttribute("data-md-color-primary");
    if (colorscheme) {
        console.log("Colorscheme changed:", colorscheme);
        localStorage.setItem("colorscheme", colorscheme);
        updateImageColorsheme();
    }
};
document.body.addEventListener("change", updateColorscheme);
document.addEventListener("DOMContentLoaded", updateColorscheme);

updateImageColorsheme();
window.addEventListener('resize', updateImageColorsheme);

