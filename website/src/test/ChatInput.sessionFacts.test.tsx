/**
 * Composer chips name what the live session reports running.
 *
 * The model chip leads with the ACP backend the session runs on and names the
 * model its harness reported, with the selection beside it in the title when
 * the two differ. The context popover repeats the backend and says whether
 * Kiro Crew's own tools reached the session, and the agent chip warns when they
 * did not: a session that kept its harness through a config switch, or that
 * runs without the gateway's MCP servers, must not look like any other.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { fireEvent, screen } from '@testing-library/react'
import i18next from 'i18next'
import { renderWithProviders } from './helpers'
import ChatInput from '../components/ChatInput'
import '../i18n/all'

vi.mock('../api/client', () => ({ api: {} }))

const props = (over: Record<string, unknown> = {}) => ({
  value: '',
  onChange: vi.fn(),
  onSend: vi.fn(),
  connected: true,
  agentName: 'cr-writer',
  onAgentClick: vi.fn(),
  onProjectClick: vi.fn(),
  onModelClick: vi.fn(),
  modelName: 'global.anthropic.claude-opus-5[1m]',
  contextPct: 12,
  ...over,
})

afterEach(async () => { await i18next.changeLanguage('en') })

describe('ChatInput — what the live session reports', () => {
  it('names the backend on the model chip and the selection in its title', () => {
    renderWithProviders(
      <ChatInput {...props({ sessionBackend: 'claude', modelSelected: 'global.anthropic.claude-fable-5[1m]' })} />,
    )
    const chip = screen.getByTestId('composer-model-chip')
    expect(screen.getByTestId('composer-model-chip-backend').textContent).toBe('Claude Code')
    // The routing prefix leaves the visible name; the title keeps the full id.
    expect(chip.textContent).toContain('claude-opus-5[1m]')
    expect(chip.textContent).not.toContain('global.anthropic.')
    const title = chip.getAttribute('title') ?? ''
    expect(title).toContain('global.anthropic.claude-opus-5[1m]')
    expect(title).toContain('Selected: global.anthropic.claude-fable-5[1m]')
    expect(title).toContain('Agent backend: Claude Code')
  })

  it('names kiro-cli, whose backend id is the empty string', () => {
    renderWithProviders(<ChatInput {...props({ sessionBackend: '' })} />)
    expect(screen.getByTestId('composer-model-chip-backend').textContent).toBe('Kiro CLI')
  })

  it('names no backend before a session reports one', () => {
    renderWithProviders(<ChatInput {...props({ sessionBackend: null })} />)
    expect(screen.queryByTestId('composer-model-chip-backend')).toBeNull()
  })

  it('warns on the agent chip and in the popover when the gateway tools were not sent', () => {
    renderWithProviders(<ChatInput {...props({ sessionBackend: 'claude', gatewayTools: 'not_sent' })} />)
    expect(screen.getByTestId('agent-chip-tools-missing')).toBeTruthy()
    const agentChip = screen.getByTestId('agent-chip-tools-missing').closest('button')!
    expect(agentChip.getAttribute('aria-label')).toContain('This session has no Kiro Crew tools')

    fireEvent.click(screen.getByRole('button', { name: 'Context usage' }))
    expect(screen.getByTestId('context-session-backend').textContent).toContain('Claude Code')
    expect(screen.getByTestId('context-gateway-tools').textContent).toContain('Not sent')
  })

  it('reports tools that failed to start as an error, not a status row', () => {
    renderWithProviders(<ChatInput {...props({ sessionBackend: '', gatewayTools: 'failed' })} />)
    expect(screen.getByTestId('agent-chip-tools-missing')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Context usage' }))
    expect(screen.queryByTestId('context-gateway-tools')).toBeNull()
    expect(screen.getByTestId('context-gateway-tools-failed').textContent).toContain(
      'Kiro Crew tools failed to start in this session.',
    )
  })

  it('does not warn when the gateway tools reached the session', () => {
    renderWithProviders(<ChatInput {...props({ sessionBackend: '', gatewayTools: 'connected' })} />)
    expect(screen.queryByTestId('agent-chip-tools-missing')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Context usage' }))
    expect(screen.getByTestId('context-gateway-tools').textContent).toContain('Connected')
  })
})
