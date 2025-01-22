require 'rouge'

module Rouge
  module Lexers
    class ASM < RegexLexer
      title "ISA"
      desc "Assembly language highlighting"

      tag 'isa'
      filenames '*.asm', '*.s'
      mimetypes 'text/x-isa'


      # Define the keywords to highlight
      BLUE_KEYWORDS = Set.new %w(ds_load_b128 ds_load_b64 ds_store_b128)
      GREEN_KEYWORDS = Set.new %w(v_lshrrev_b32_e32 s_add_u32 s_addc_u32 s_lshl_b32 v_lshlrev_b32_e32 v_add_u32_e32 v_and_b32_e32 v_add_co_ci_u32_e32 v_add_co_u32 v_lshlrev_b64 v_ashrrev_i32_e32 v_add_nc_u32_e32 v_fmac_f32_e32 v_add_f32 v_mul_f32 v_dual_fmac_f32 v_mov_b32)
      RED_KEYWORDS = Set.new %w(s_waitcnt)
      PURPLE_KEYWORDS = Set.new %w(global_load_b32)

 
      state :root do
  

        # Highlight ds_load_b128 and ds_store_b128 in blue
        rule %r/\b(#{BLUE_KEYWORDS.to_a.join('|')})\b/, Keyword::Type
        # Highlight v_add_f32, v_mul_f32, v_dual_fmac_f32 in green
        rule %r/\b(#{GREEN_KEYWORDS.to_a.join('|')})\b/, Keyword::Constant
        # Highlight s_waitcnt in red
        rule %r/\b(#{RED_KEYWORDS.to_a.join('|')})\b/, Keyword::Reserved
        rule %r/\b(#{PURPLE_KEYWORDS.join('|')})\b/, Name::Class    # Purple
        rule %r/::/ do
            token Operator
            reset_stack
          end
        # Registers (v0, v1, etc.) — use Name for general variable highlighting
        rule %r/\b(v|s|t)\d+\b/, Name  # Using Name (generic for variables)
        # Numbers
        rule %r/\b\d+\b/, Num
        # Comments
        rule %r/;.*$/, Comment::Single
        # Punctuation (e.g., commas, semicolons)
        rule %r/[;,]/, Punctuation
        # Whitespace
        rule %r/\s+/, Text
        # Default: Any other text (fallback)
        rule %r/.+/, Text # Match non-empty text as the fallback
      end
    end
  end
end
