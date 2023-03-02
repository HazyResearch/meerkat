<script>
    {% if import_style == 'named' -%}
    import {{ component_name }} from '{{ path }}';
    {% elif import_style == 'none' -%}
    {% else -%}
    import { {{ component_name }} } from '{{ path }}';
    {% endif %}
    {% if event_names -%}
    {% if is_user_app -%}
    import { dispatch } from '@meerkat-ml/meerkat/utils/api';
    {% else -%}
    import { dispatch } from '$lib/utils/api';
    {% endif -%}
    {% endif -%}
    import { setContext } from 'svelte';

    {% for prop in prop_names -%}
    export let {{ prop }};
    {% endfor -%}
    {% for event in event_names -%}
    export let on_{{ event }};
    {% endfor -%}

    export let component_id;
    if (component_id) {
        // some components do not have a component_id, e.g.Cells, because they are 
        // not created on the Python side. These should not overrride the context
        // of the parent component.
        setContext('componentId', $component_id);
    }
</script>

<{{ component_name }} 
    {% for prop, prop_camel_case in zip(prop_names, prop_names_camel_case) -%}
    {% if prop_bindings[prop] -%}
    {% if prop == 'classes' and import_style == 'none' -%}
    {# For HTML tags, convert the classes attribute to the class property #}
    class={${{ prop }}}
    {% elif prop == 'classes' and path == 'flowbite-svelte' -%}
    {# For Flowbite-Svelte, convert the classes attribute to the class property #}
    class={${{ prop }}}
    {% elif import_style == 'none' -%}
    {# For HTML tags, do not use bind, and convert _ to -, e.g. aria_hidden to aria-hidden #}
    {{ prop.replace("_", "-") }}={${{ prop }}}
    {% else -%}
    bind:{{ prop_camel_case }}={${{ prop }}}
    {% endif -%}
    {% else -%}
    {{ prop_camel_case }}={ {{prop}} }
    {% endif -%}
    {% endfor -%}
    {% for event in event_names -%}
    on:{{ event }}={(e) => on_{{ event }} ? dispatch(on_{{ event }}.endpointId, {detail: e.detail}) : null}
    {% endfor -%}
>
    {% if slottable -%}
    <slot />
    {% endif %}
</{{ component_name }}>