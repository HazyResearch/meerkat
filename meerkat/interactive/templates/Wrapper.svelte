<script>
    {% if import_style == 'named' -%}
    import {{ component_name }} from '{{ path }}';
    {% elif import_style == 'none' -%}
    {% else -%}
    import { {{ component_name }} } from '{{ path }}';
    {% endif %}
    {% if event_names -%}
    import { getContext } from 'svelte';
    const { dispatch } = getContext('Meerkat');
    {% endif -%}

    {% for prop in prop_names -%}
    export let {{ prop }};
    {% endfor -%}
    {% for event in event_names -%}
    export let on_{{ event }};
    {% endfor -%}
</script>

<{{ component_name }} 
    {% for prop in prop_names -%}
    {% if prop_bindings[prop] -%}
    {% if prop == 'classes' and import_style == 'none' -%}
    {# For HTML tags, convert the classes attribute to the class property #}
    class={${{ prop }}}
    {% elif import_style == 'none' -%}
    {# For HTML tags, do not use bind, and convert _ to -, e.g. aria_hidden to aria-hidden #}
    {{ prop.replace("_", "-") }}={${{ prop }}}
    {% else -%}
    bind:{{ prop }}={${{ prop }}}
    {% endif -%}
    {% else -%}
    {{ prop }}={ {{prop}} }
    {% endif -%}
    {% endfor -%}
    {% for event in event_names -%}
    on:{{ event }}={(e) => on_{{ event }} ? dispatch(on_{{ event }}.endpoint_id, {detail: e.detail}) : null}
    {% endfor -%}
>
    {% if slottable -%}
    <slot />
    {% endif %}
</{{ component_name }}>