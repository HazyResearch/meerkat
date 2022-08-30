import { global_stores, store_lock } from "$lib/components/blanks/stores";
import { get as get_store } from "svelte/store";


export async function get(url: string): Promise<any> {
    const res: Response = await fetch(url);
    if (!res.ok) {
        throw new Error('HTTP status ' + res.status);
    }
    const json = await res.json();
    return json;
}

export async function post(url: string, data: any): Promise<any> {
    const res: Response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        throw new Error(
            "HTTP status " + res.status + ": " + res.statusText + "\n url: " + url + "\n data: " + JSON.stringify(data)
        );
    }
    const json = await res.json();
    return json;
}


export async function modify(url: string, data: any): Promise<any> {
    let modifications = await post(url, data);
    console.log(modifications)

    // url must hit an endpoint that returns a list of modifications 
    for (let modification of modifications) {
        if (modification.type === 'box') {
            // Box modification
            let store = global_stores.get(modification.id)
            store.update((value: any) => {
                value.scope = modification.scope;
                return value
            }
            )
        } else if (modification.type === 'store') {
            // Store modification
            get_store(store_lock).add(modification.id);
            let store = global_stores.get(modification.id);
            store.set(modification.value);
        }
    }
}