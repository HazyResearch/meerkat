import { writable } from 'svelte/store';
import { browser } from "$app/env";

export const api_url = writable('');

// https://dev.to/delanyobott/comment/1egh0
if (browser) {
    api_url.set(localStorage.getItem("api_url"));
    api_url.subscribe(value => {
        localStorage.setItem("api_url", value ? value : '');
    });
}
