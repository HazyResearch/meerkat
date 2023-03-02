import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import Message from './Message.svelte';


describe('Message', () => {
    it('should render message', () => {
        const { container } = render(Message, {
            props: {
                message: 'Hello world!',
                name: 'John Doe',
                time: '12:00',
            },
        });
        expect(container).toMatchSnapshot();

        // Expect the message to exist in the p tag
        const p = screen.getByText('Hello world!');
        expect(p.textContent).toContain('Hello world!');
    });
});
