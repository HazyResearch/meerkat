<script lang="ts">
    import { setContext } from 'svelte/internal';
    
    {% for component in frontend_components -%}
    import {{ component.frontend_alias }} from '$lib/wrappers/__{{ component.namespace }}/{{ component.component_name }}.svelte';
    {% endfor %}
    
    $: setContext("Components", {
        {% for component in components -%}
        "{{ component.alias }}": {{ component.frontend_alias }},
        {% endfor %}
    })
</script>

<slot />
