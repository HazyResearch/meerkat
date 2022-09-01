<script lang="ts">
	export let data: any;
	export let dtype: string | undefined = undefined;

    if(dtype === undefined){
        if(typeof data === "number") {
            
            if(Number.isInteger(data)) {
                dtype = "int";
            } else {
                dtype = "float";
            }
        } else {
            dtype = "string"
        }
    }

    
    if (dtype === "float") {
        data = data.toPrecision(3);
    }
</script>

{data}
