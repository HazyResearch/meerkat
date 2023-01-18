<script>
    {% if import_style == 'named' -%}
    import {{ component_name }} from '{{ path }}';
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
    {% if use_bindings %}
    {% if prop_bindings[prop] -%}
    bind:{{ prop }}={${{ prop }}}
    {% else -%}
    {{ prop }}={ {{prop}} }
    {% endif -%}
    {% else -%}
    bind:{{ prop }}={${{ prop }}}
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