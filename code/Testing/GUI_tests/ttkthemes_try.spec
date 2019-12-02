# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['ttkthemes_try.py'],
             pathex=['C:\\Users\\Robin\\Documents\\GitHub\\ESAT7A1\\code\\Testing\\GUI_tests'],
             binaries=[],
             datas=[('venv\\Lib\\site-packages\\ttkthemes', 'ttkthemes')],
             hiddenimports=['ttkthemes'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='ttkthemes_try',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ttkthemes_try')
