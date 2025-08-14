window.addEventListener('message', (event) => {
    if (event.data.type === 'resize') {
        console.log("here", event.data.height);
        document.getElementById(
            'iframe-demo-0'
        ).style.height = event.data.height + 'px';
    }
});
