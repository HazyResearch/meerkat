<script lang="ts">
	export let data: any;
	export let dtype: string | undefined = undefined;
    export let precision: number = 3
    export let percentage: boolean = false

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
        if (percentage) {
            data = (data * 100).toPrecision(precision) + "%";
        } else {
            data = data.toPrecision(precision);
        }
    }
</script>

{data}