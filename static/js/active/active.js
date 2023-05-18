const listRoute = document.querySelectorAll('.app-menu__item');
const path = window.location.pathname;
const currentPath = path.split("/");

listRoute.forEach((route) => {
    if (route.getAttribute('href') && currentPath[1]) {
        if (route.getAttribute('href').includes(currentPath[1])) {
            route.classList.add('active');
        } else {
            route.classList.remove('active');
        }
    }
    
});