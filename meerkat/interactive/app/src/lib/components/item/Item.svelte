<script context="module" lang="ts">
    export type ItemType = string | number | boolean;
</script>    

<script lang="ts">
    export let data: ItemType;
    export let formatter = (value: ItemType, type: string) => {
        if (type === 'number' || type === 'boolean') {
            return value.toString();
        } else if (type === 'image') {
            return `<img class="block object-center h-4/5 w-full rounded-md shadow-lg" src="${value}"/>`;
        } else {
            return value as string;
        }
    };

    let infer_type = (value: ItemType) => {
        if (typeof value === 'string') {
            if (value.startsWith('data:image') || value.startsWith('http')) {
                return 'image';
            } else {
                return 'string';
            }
        } else {
            return typeof value;
        }
    };

    let datatype = infer_type(data);
    
</script>

{@html formatter(data, datatype)}
