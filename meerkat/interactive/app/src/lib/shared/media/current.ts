import { writable, type Writable } from 'svelte/store';

export class PlayingMedia {
    data: any;
    currentTime: number = 0;
    duration: number = 0;
    paused: boolean = true;
    ended: boolean = false;


    constructor(data: any) {
        this.data = data;
    }
}

export const currentMedia: Writable<PlayingMedia | null> = writable(null);


export const setCurrentMedia = (id: string) => {
    let media = new PlayingMedia(id);
    currentMedia.set(media);
    return media;
};