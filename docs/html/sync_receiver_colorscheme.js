let lastColorsheme = null;
const applyColorscheme = () => {
    var palette = localStorage.getItem("colorscheme");
    if (palette && palette !== lastColorsheme) {
        lastColorsheme = palette;

        console.log("Applying colorscheme:", palette);
        if (palette.startsWith('black')) {
            document.documentElement.classList.add("theme-dark");
        } else if (palette.startsWith('white')) {
            document.documentElement.classList.remove("theme-dark");
        } else {
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                document.documentElement.classList.add("theme-dark");
            } else {
                document.documentElement.classList.remove("theme-dark");
            }
        }
    }

    console.log("Starting colorscheme sync receiver...");
};

addEventListener("storage", applyColorscheme);
document.addEventListener("DOMContentLoaded", applyColorscheme);
