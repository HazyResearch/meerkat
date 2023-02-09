import { globalStores } from "$lib/utils/stores";
import { get } from "svelte/store";


export async function getRequest(url: string): Promise<any> {
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
    const json = await res.json();
    if (!res.ok) {
        console.log(
            "HTTP status " + res.status + ": " + res.statusText + "\n url: " + url + "\n data: " + JSON.stringify(data)
        );
        throw new Error("HTTP status " + res.status + ": " + json.detail)
    }
    return json;
}

export async function modify(url: string, data: any): Promise<any> {
    const modifications = await post(url, data);
    return applyModifications(modifications);
}

export function applyModifications(modifications: Array<any>) {
    for (const modification of modifications) {
        if (modification.type === 'ref') {
            // Node modification
            if (!globalStores.has(modification.id)) {
                // derived objects may not be maintained on the frontend 
                // TODO: consider adding a mechanism to add new derived objects to the frontend
                continue
            }
            const store = globalStores.get(modification.id)
            store.update((value: any) => {
                value.scope = modification.scope;
                return value
            }
            )
        } else if (modification.type === 'store') {
            // Store modification
            const store = globalStores.get(modification.id);

            if (store === undefined) {
                console.log(
                    "Store is not maintained on the frontend. Only stores passed as props to a component are maintained in `globalStores`.",
                    modification.id
                )
                continue;
            }
            // set with trigger=false so that the store change doesn't trigger backend 
            if (!("triggerStore" in store)) {
                throw "Must use `meerkatWritable` for backend stores."
            }
            // only update the store if the value has changed
            if (modification.value !== get(store)) {
                store.set(modification.value, false);
            }
        }
    }
}