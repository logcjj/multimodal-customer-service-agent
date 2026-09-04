import { fireEvent, render, screen } from '@testing-library/react';
import type { ConversationSummary } from '../api/types';
import { ConversationSidebar } from './conversation-sidebar';

const now = new Date();
const items: ConversationSummary[] = [
  {
    id: 'c1',
    owner_id: 'owner-a',
    title: '洗衣机 E03 排障',
    message_count: 2,
    last_message_preview: '请先断电',
    last_route: 'technical_knowledge',
    created_at: now.toISOString(),
    updated_at: now.toISOString(),
  },
];

describe('ConversationSidebar', () => {
  beforeAll(() => {
    window.PointerEvent = MouseEvent as typeof PointerEvent;
  });

  it('supports new, active selection, rename and confirmed delete', () => {
    const onNew = jest.fn();
    const onSelect = jest.fn();
    const onRename = jest.fn();
    const onDelete = jest.fn();
    render(
      <ConversationSidebar
        items={items}
        activeId="c1"
        disabled={false}
        onNew={onNew}
        onSelect={onSelect}
        onRename={onRename}
        onDelete={onDelete}
      />,
    );

    expect(screen.getByText('今天')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '洗衣机 E03 排障' })).toHaveAttribute(
      'aria-current',
      'true',
    );
    expect(
      screen.getByRole('button', { name: '更多操作：洗衣机 E03 排障' }),
    ).toHaveClass(
      'opacity-100',
      'xl:opacity-0',
      'xl:focus-visible:opacity-100',
    );
    fireEvent.click(screen.getByRole('button', { name: '新建对话' }));
    expect(onNew).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: '洗衣机 E03 排障' }));
    expect(onSelect).toHaveBeenCalledWith('c1');

    fireEvent.pointerDown(
      screen.getByRole('button', { name: '更多操作：洗衣机 E03 排障' }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(screen.getByText('重命名'));
    const titleInput = screen.getByRole('textbox', { name: '对话标题' });
    fireEvent.change(titleInput, { target: { value: '排水故障复查' } });
    fireEvent.keyDown(titleInput, { key: 'Enter' });
    expect(onRename).toHaveBeenCalledWith('c1', '排水故障复查');

    fireEvent.pointerDown(
      screen.getByRole('button', { name: '更多操作：洗衣机 E03 排障' }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(screen.getByText('删除'));
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }));
    expect(onDelete).toHaveBeenCalledWith('c1');
  });

  it('disables all conversation mutations while an answer is running', () => {
    render(
      <ConversationSidebar
        items={items}
        activeId="c1"
        disabled
        onNew={jest.fn()}
        onSelect={jest.fn()}
        onRename={jest.fn()}
        onDelete={jest.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: '新建对话' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '洗衣机 E03 排障' })).toBeDisabled();
    expect(
      screen.getByRole('button', { name: '更多操作：洗衣机 E03 排障' }),
    ).toBeDisabled();
  });
});
