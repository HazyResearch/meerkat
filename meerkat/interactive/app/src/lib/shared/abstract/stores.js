import { derived, writable } from "svelte/store";

export const state = writable({});
export const interfaces = writable({});
export const blocks = writable({}); 


export const block_groups = derived(
    [blocks, interfaces],
    ($blocks, $interfaces) => {
        const _block_groups = {};
        for (const block_id in blocks) {
            // Get the base_dataframe_id for this block
            const block = $blocks[block_id];
            const group_id = `${block.interface_id}-${block.base_dataframe_id}`;
            if (!(group_id in block_groups)) {
                _block_groups[group_id] = {
                    group_id,
                    blocks: [],
                };
            }
            _block_groups[group_id].blocks.push(block_id);
        }
        return _block_groups;
    }
    );

export const application = writable({
    "interfaces": {},
    "blocks": {},
    "block_groups": {},
})